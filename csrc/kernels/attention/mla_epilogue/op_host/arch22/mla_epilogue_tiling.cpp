/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_epilogue_tiling.cpp
 * \brief mla_epilogue Host 侧 Tiling 实现（arch22 / DAV_2201，DESIGN §3.4）
 *
 * 方案要点（严格按 DESIGN §3.4）：
 *  - 所有平台能力（核数 / UB 容量 / 系统 workspace）运行时获取，禁止写死 24 / 48 / 192 KB；
 *  - 单核 matmul_tiling::MatmulApiTiling 生成两套 TCubeTiling，N 分核与尾块由 Host 手工推导，
 *    Kernel 侧以 SetSingleShape + 偏移指针下发；不复写 usedCoreNum（P7）；
 *  - GEMM1：B 非转置（Host SetBType isTrans=false），GEMM2：B 转置（isTrans=true），
 *    与 Kernel 的 MatmulType ISTRANS、运行时 SetTensorB 三处一致（P6）；
 *  - Matmul 片上临时空间用 MatmulGetTmpBufSizeV2 实测（M-10 口径），不用 shareUbSize 预留字段；
 *  - 系统 workspace 用 GetLibApiWorkSpaceSize()，与 Kernel 的 GetSysWorkSpacePtr() 配套（P1）。
 */

#include <string>
#include <vector>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/tiling_api.h"
#include "base/err_msg.h"
#include "../../op_kernel/arch22/mla_epilogue_tiling_data.h"
#include "../../op_kernel/arch22/mla_epilogue_tiling_key.h"

namespace optiling {

constexpr size_t WORKSPACE_NUM = 1U;

// RoPE UB 口径（DESIGN §3.4 第 3 步 / §3.9 / §3.10）
constexpr uint64_t UB_RESERVED_SIZE = 8192ULL;      // 框架预留
constexpr uint64_t ROPE_ROW_UB = 16384ULL;          // 每行双缓冲 xTile：2 × 4096 × 2 B
constexpr uint64_t ROPE_FIXED_UB = 5888ULL;         // ropeF32 + swapF32 + cos/sin 双缓冲 + 3 个辅助向量
// 设备内 ND→NZ 预转换（大 S 门控）：AIV 用 UB 暂存按 32 B 块做跨行搬运。
constexpr uint32_t NZ_CONV_MIN_S = 8192U;           // 只有 S >= 该值才做转换（转换成本 ~468us）
constexpr uint64_t NZ_STAGE_UB = 32768ULL;          // 2 × 16 KiB 双缓冲暂存（v1）
constexpr uint32_t TILE_S_CAP = 8U;                 // 设计余量封顶
constexpr uint32_t WS_ALIGN = 512U;

// shape 固定维度（spec shape_constraints）
constexpr int64_t DIM_X_S_HEAD = 64;
constexpr int64_t DIM_X_D = 512;
constexpr int64_t DIM_ROPE = 64;
constexpr int64_t MLA_MAX_S = 131072;

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

static inline uint32_t AlignUpU32(uint32_t a, uint32_t align)
{
    return (a + align - 1U) / align * align;
}

static inline uint64_t AlignUpU64(uint64_t a, uint64_t align)
{
    return (a + align - 1U) / align * align;
}

// =============================================================================
// 输入校验（DESIGN §3.4 第 1 步；对应 spec dtype_policy / shape_constraints / boundary_conditions）
// =============================================================================

// 目标版本 op_common/log/log.h 的 Ops::Base::ToString 实现位于 libops_base.so，
// 而本算子 tiling 库的链接链（libnnopbase / libregister / libtiling_api）未引入该库，
// 直接引用会留下加载期无法解析的强符号，故在本文件内实现等价的字符串拼接。
static std::string DataTypeToString(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return "FLOAT";
        case ge::DT_FLOAT16:
            return "FLOAT16";
        case ge::DT_BF16:
            return "BF16";
        case ge::DT_INT8:
            return "INT8";
        case ge::DT_INT16:
            return "INT16";
        case ge::DT_INT32:
            return "INT32";
        case ge::DT_INT64:
            return "INT64";
        case ge::DT_UINT8:
            return "UINT8";
        case ge::DT_DOUBLE:
            return "DOUBLE";
        case ge::DT_BOOL:
            return "BOOL";
        default:
            return "DT_" + std::to_string(static_cast<int32_t>(dataType));
    }
}

static std::string ShapeToString(const gert::Shape& shape)
{
    std::string result = "[";
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
        if (i != 0U) {
            result += ",";
        }
        result += std::to_string(shape.GetDim(i));
    }
    result += "]";
    return result;
}

static std::string DimToString(size_t dimNum)
{
    return std::to_string(dimNum) + "D";
}

// 合法输入下的 Tiling / 平台 / 库调用失败：由根因层首报一次 EZ1008（Arglist: opName, reason），
// 调用点负责构造包含原始返回码与关键入参的可操作诊断。
static ge::graphStatus ReportTilingFailure(gert::TilingContext* context, const std::string& reason)
{
    OP_LOGE_WITHOUT_REPORT(context, "%s", reason.c_str());
    const std::string opInfo = Ops::Base::GetOpInfo(context);
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opInfo.c_str(), reason.c_str()};
    (void)error_message::ReportPredefinedErrMsg("EZ1008", msgKey, msgValue);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus CheckInputDtypes(gert::TilingContext* context)
{
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const gert::CompileTimeTensorDesc* cosDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosDesc);
    const gert::CompileTimeTensorDesc* sinDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinDesc);
    const gert::CompileTimeTensorDesc* w1Desc = context->GetInputDesc(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, w1Desc);
    const gert::CompileTimeTensorDesc* w2Desc = context->GetInputDesc(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, w2Desc);
    const gert::CompileTimeTensorDesc* yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    const std::string opInfo = Ops::Base::GetOpInfo(context);
    const ge::DataType xDt = xDesc->GetDataType();
    if (xDt != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "x", DataTypeToString(xDt), "BF16");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType cosDt = cosDesc->GetDataType();
    if (cosDt != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "cos", DataTypeToString(cosDt), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType sinDt = sinDesc->GetDataType();
    if (sinDt != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "sin", DataTypeToString(sinDt), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType w1Dt = w1Desc->GetDataType();
    if (w1Dt != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "w1", DataTypeToString(w1Dt), "BF16");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType w2Dt = w2Desc->GetDataType();
    if (w2Dt != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "w2", DataTypeToString(w2Dt), "BF16");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType yDt = yDesc->GetDataType();
    if (yDt != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "y", DataTypeToString(yDt), "BF16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShapes(gert::TilingContext* context, uint32_t& totalS)
{
    const std::string opInfo = Ops::Base::GetOpInfo(context);

    // x: [S, 1, 64, 512]
    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape& xShape = xStorageShape->GetStorageShape();
    if (xShape.GetDimNum() != 4U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opInfo, "x", DimToString(xShape.GetDimNum()), "4D");
        return ge::GRAPH_FAILED;
    }
    if (xShape.GetDim(1) != 1 || xShape.GetDim(2) != DIM_X_S_HEAD || xShape.GetDim(3) != DIM_X_D) {
        OP_LOGE_FOR_INVALID_SHAPE(opInfo, "x", ShapeToString(xShape), "[S,1,64,512]");
        return ge::GRAPH_FAILED;
    }
    const int64_t s = xShape.GetDim(0);
    if (s < 1 || s > MLA_MAX_S) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opInfo, "x", std::to_string(s),
                                             "S must be in [1,131072]");
        return ge::GRAPH_FAILED;
    }

    // cos / sin: [S, 1, 1, 64]
    const char* const cosSinNames[] = {"cos", "sin"};
    for (size_t i = 1U; i <= 2U; ++i) {
        const gert::StorageShape* csStorageShape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, csStorageShape);
        const gert::Shape& csShape = csStorageShape->GetStorageShape();
        if (csShape.GetDimNum() != 4U) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opInfo, cosSinNames[i - 1U], DimToString(csShape.GetDimNum()), "4D");
            return ge::GRAPH_FAILED;
        }
        if (csShape.GetDim(1) != 1 || csShape.GetDim(2) != 1 || csShape.GetDim(3) != DIM_ROPE) {
            OP_LOGE_FOR_INVALID_SHAPE(opInfo, cosSinNames[i - 1U], ShapeToString(csShape), "[S,1,1,64]");
            return ge::GRAPH_FAILED;
        }
        if (csShape.GetDim(0) != s) {
            const std::string paramNames = "x," + std::string(cosSinNames[i - 1U]);
            const std::string incorrectShapes = ShapeToString(xShape) + "," + ShapeToString(csShape);
            const std::string reason =
                "the S dimension of " + std::string(cosSinNames[i - 1U]) + " must equal the S dimension of x";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opInfo, paramNames, incorrectShapes, reason);
            return ge::GRAPH_FAILED;
        }
    }

    // w1: [8, 4096, 1024]
    const gert::StorageShape* w1StorageShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, w1StorageShape);
    const gert::Shape& w1Shape = w1StorageShape->GetStorageShape();
    if (w1Shape.GetDimNum() != 3U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opInfo, "w1", DimToString(w1Shape.GetDimNum()), "3D");
        return ge::GRAPH_FAILED;
    }
    if (w1Shape.GetDim(0) != static_cast<int64_t>(MLA_G) ||
        w1Shape.GetDim(1) != static_cast<int64_t>(MLA_K1) ||
        w1Shape.GetDim(2) != static_cast<int64_t>(MLA_N1)) {
        OP_LOGE_FOR_INVALID_SHAPE(opInfo, "w1", ShapeToString(w1Shape), "[8,4096,1024]");
        return ge::GRAPH_FAILED;
    }

    // w2: [5120, 8192]
    const gert::StorageShape* w2StorageShape = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, w2StorageShape);
    const gert::Shape& w2Shape = w2StorageShape->GetStorageShape();
    if (w2Shape.GetDimNum() != 2U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opInfo, "w2", DimToString(w2Shape.GetDimNum()), "2D");
        return ge::GRAPH_FAILED;
    }
    if (w2Shape.GetDim(0) != static_cast<int64_t>(MLA_N2) ||
        w2Shape.GetDim(1) != static_cast<int64_t>(MLA_K2)) {
        OP_LOGE_FOR_INVALID_SHAPE(opInfo, "w2", ShapeToString(w2Shape), "[5120,8192]");
        return ge::GRAPH_FAILED;
    }

    totalS = static_cast<uint32_t>(s);
    return ge::GRAPH_SUCCESS;
}


// =============================================================================
// MDL 大块搬运调度参数（性能优化 r2）
// =============================================================================
// Matmul 走 CFG_MDL 调度时，每个外层迭代按 stepKa/stepKb 个 baseK 块一次性搬 A/B：
//   A tile = baseM × (stepKa×baseK)，B tile = baseN × (stepKb×baseK)。
// 库默认（GEMM1 stepKa=16/stepKb=1，GEMM2 stepKa=2/stepKb=8）在本算子的 N 手工分核 +
// 单 baseN tile 场景下，GEMM1 的 B 搬运粒度过小（256×1×64×2 = 32 KiB/条），MTE2 指令数
// 偏高（S=96 实测全核 3,264 条，平均 31 KiB/条）。把 stepKb 提高到 2（GEMM1）/ 4
// （GEMM2，等价更大的 B tile），并把 depthA1/depthB1 按 2×stepK 重新配套（保持双缓冲），
// 单条 GM→L1 搬运粒度提升到 64 KiB / 114 KiB，MTE2 指令数减半，S=96 op-level 由
// 181.0 us 降到 ~169 us（21 会话同协议）。
// 约束（与库的校验规则一致）：
//   kIter = K / baseK 必须能被 stepKa/stepKb 整除；
//   stepN=1（kIter > stepKb 恒成立）时 depthB1 % stepKb == 0 且 ratio ≤ 2；
//   stepM=1 时 depthA1 % stepKa == 0 且 ratio ≤ 2；
//   A+B 的 L1 双缓冲占用 ≤ L1 容量（否则保留库 tiling，避免越界）。
static bool ApplyMdlStepTiling(AscendC::tiling::TCubeTiling& t, int32_t stepKa, int32_t stepKb,
                              uint64_t l1Size)
{
    if (t.baseM <= 0 || t.baseN <= 0 || t.baseK <= 0 || t.Ka <= 0) {
        return false;
    }
    const int32_t kIter = t.Ka / t.baseK;
    if (kIter <= 0 || (kIter % stepKa) != 0 || (kIter % stepKb) != 0) {
        return false;
    }
    if (t.stepM > 1 || t.stepN > 1) {
        return false; // 本算子两段 GEMM 恒为 stepM=stepN=1；其他形态不接管
    }
    const int32_t depthA = 2 * stepKa;
    const int32_t depthB = 2 * stepKb;
    const uint64_t needA = static_cast<uint64_t>(depthA) * static_cast<uint64_t>(t.baseM) *
                           static_cast<uint64_t>(t.baseK) * 2ULL;
    const uint64_t needB = static_cast<uint64_t>(depthB) * static_cast<uint64_t>(t.baseN) *
                           static_cast<uint64_t>(t.baseK) * 2ULL;
    const uint64_t margin = 16ULL * 1024ULL; // 预留余量，避免与库内部其他占用相撞
    if (needA + needB + margin > l1Size) {
        return false;
    }
    t.stepKa = stepKa;
    t.depthA1 = depthA;
    t.stepKb = stepKb;
    t.depthB1 = depthB;
    return true;
}

// =============================================================================
// Tiling 主体
// =============================================================================

static ge::graphStatus TilingMlaEpilogue(gert::TilingContext* context)
{
    // ---- 1. 平台信息（运行时获取，DC-6）----
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    platform_ascendc::PlatformAscendC platform(platformInfoPtr);
    const uint32_t aicNum = platform.GetCoreNumAic();
    const uint32_t aivNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0ULL;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (aicNum == 0U) {
        return ReportTilingFailure(context, "GetCoreNumAic() returned 0");
    }
    if (aivNum == 0U) {
        return ReportTilingFailure(context, "GetCoreNumAiv() returned 0");
    }
    if (ubSize == 0U) {
        return ReportTilingFailure(context, "GetCoreMemSize(UB) returned 0");
    }

    // ---- 2. 输入校验 ----
    // 参数错误已在 CheckInputDtypes / CheckInputShapes 内部完成正式首报，此处仅记录上下文并透传。
    ge::graphStatus checkStatus = CheckInputDtypes(context);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "CheckInputDtypes failed, rc=%u", static_cast<unsigned int>(checkStatus));
        return checkStatus;
    }
    uint32_t totalS = 0U;
    checkStatus = CheckInputShapes(context, totalS);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "CheckInputShapes failed, rc=%u", static_cast<unsigned int>(checkStatus));
        return checkStatus;
    }

    // ---- 3. TilingData ----
    MlaEpilogueTilingData* tiling = context->GetTilingData<MlaEpilogueTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const errno_t memsetRc = memset_s(tiling, sizeof(MlaEpilogueTilingData), 0, sizeof(MlaEpilogueTilingData));
    if (memsetRc != EOK) {
        return ReportTilingFailure(context, "memset_s(MlaEpilogueTilingData) failed, errno=" + std::to_string(memsetRc));
    }

    // ---- 4. M 对齐与流水段划分 ----
    const uint32_t paddedM = AlignUpU32(totalS, MLA_ALIGN16);
    const uint32_t padRows = paddedM - totalS;

    // RoPE 每 tile 行数：先算容量上限（§3.4 第 3 步），容量公式与 §3.9/§3.10 统一
    const uint64_t usableUB = (ubSize > UB_RESERVED_SIZE) ? (ubSize - UB_RESERVED_SIZE) : 0ULL;
    uint32_t tileSMax = 1U;
    if (usableUB > ROPE_FIXED_UB) {
        const uint64_t cap = (usableUB - ROPE_FIXED_UB) / ROPE_ROW_UB;
        tileSMax = static_cast<uint32_t>(cap < 1ULL ? 1ULL : (cap > TILE_S_CAP ? TILE_S_CAP : cap));
    }

    // M-tile 流水段划分（性能优化：单段）：
    //   原设计 ptn=2 让 RoPE(t+1) 与 GEMM1(t) 重叠，但每个流水段都要完整遍历一遍 w1：
    //   实测（S=96）GEMM1 阶段 AIC 的 L2 读请求量为 180.0 MB（msopprof Memory.read_main_memory，
    //   128 B 请求粒度计数），而 w1 唯一数据量仅 67.1 MB（=65,536 KB）——第二段对 w1 的请求
    //   再次穿到 L2/HBM，违反 REQUIREMENTS §4.8.1「W1/W2 各只读一次」。
    //   改为单段（ptn=1）后每核只读一遍自己那份 w1（全核合计 = 65,536 KB）。代价是 RoPE 回到
    //   串行：全部 RoPE 在 B0 之前的首段完成，不再与 GEMM1 重叠；S=96 实测 RoPE 串行暴露仅
    //   2.44 us（阶段隔离：lead-RoPE+GEMM1 114.50 vs GEMM1 112.06），远小于 w1 单趟化省下的
    //   58.8 us。整核（S=96）296.7 → 209.2 us（msopprof op-level 21 会话中位数）。
    const uint32_t pipeTileNum = 1U;
    const uint32_t pipeTileRows = paddedM;

    // 核利用率收缩：单段任务空间需覆盖全部 AIV
    const uint32_t sTilesNeeded = CeilDivU32(aivNum, MLA_G);
    uint32_t tileS = pipeTileRows / sTilesNeeded;
    if (tileS < 1U) {
        tileS = 1U;
    }
    if (tileS > tileSMax) {
        tileS = tileSMax;
    }
    const uint32_t sTileNum = CeilDivU32(pipeTileRows, tileS);
    // Row-major RoPE：每个任务处理一整行的全部 8 个 group，cos/sin 每行只加载一次。
    const uint32_t ropeTaskNum = pipeTileRows;
    const uint32_t ropeTasksPerCore = CeilDivU32(ropeTaskNum, aivNum);
    const uint32_t aivUsedCores = CeilDivU32(ropeTaskNum, ropeTasksPerCore);

    // 一行覆盖全部 group（32768 bf16 = 64 KiB），双缓冲与原 tileS=8 的峰值相同。
    const uint64_t ropeUB = static_cast<uint64_t>(TILE_S_CAP) * ROPE_ROW_UB + ROPE_FIXED_UB;
    if (ropeUB > usableUB) {
        return ReportTilingFailure(context, "RoPE UB peak " + std::to_string(ropeUB) + " > usableUB " +
                                                 std::to_string(usableUB) +
                                                 " (tileS=" + std::to_string(tileS) + ")");
    }

    // ---- 4b. 设备内 ND→NZ 预转换开关（按 S 门控；小 S 保持 ND 路径，行为不变）----
    const bool nzConv = (totalS >= NZ_CONV_MIN_S);
    if (nzConv && (ropeUB + NZ_STAGE_UB > usableUB)) {
        return ReportTilingFailure(context, "NZ convert UB peak " + std::to_string(ropeUB + NZ_STAGE_UB) +
                                                 " > usableUB " + std::to_string(usableUB));
    }

    // ---- 5. GEMM1 分核（只切 N，权重只读一次）----
    uint32_t nSplit = aicNum / MLA_G;
    if (nSplit < 1U) {
        nSplit = 1U; // 目标平台不成立（DC-7）；Kernel 用 aicUsedCores/nSplit 作为 g 步长兜底
    }
    const uint32_t n1Units = MLA_N1 / MLA_ALIGN16; // 64
    // 128 B 对齐分核（性能优化）：MTE2 的 GM→L1 请求粒度为 128 B。若各核 N 块起点只落在
    // 64 B 边界（均匀切分为 352/336/336 元素 → 704/672/672 B），除首核外每个 K 行读都会跨
    // sector，实测非首核 MTE2 时间高出约 1.5×（121 us vs 82 us）。故把每核单元数收成
    // GEMM1_UNIT_ALIGN 个 16 元素单元（64 元素 = 128 B）的整数倍，余数全部给最后一个核：
    //   nSplit=3, n1Units=64 → base=20（320 元素）, last=24（384 元素），n0 = 0/320/640 元素
    // 全程无 64 B 边界起点，singleNMax = 384（320/320/384）。
    const uint32_t gemm1UnitAlign = 4U; // 4 × 16 = 64 元素 = 128 B
    uint32_t base1 = n1Units;
    uint32_t last1 = n1Units;
    if (nSplit > 1U) {
        const uint32_t baseAligned = ((n1Units / nSplit) / gemm1UnitAlign) * gemm1UnitAlign;
        // 兜底：平均单元数不足一个对齐单元时退回均匀切分（正确性不受影响，仅损失对齐收益）
        base1 = (baseAligned > 0U) ? baseAligned : (n1Units / nSplit);
        last1 = n1Units - base1 * (nSplit - 1U);
    }
    if ((base1 == 0U) || (base1 * (nSplit - 1U) + last1 != n1Units)) {
        return ReportTilingFailure(context, "GEMM1 N split invalid (nSplit=" + std::to_string(nSplit) +
                                                 " base=" + std::to_string(base1) +
                                                 " last=" + std::to_string(last1) +
                                                 " units=" + std::to_string(n1Units) + ")");
    }
    const uint32_t gemm1SingleNMax = MLA_ALIGN16 * ((base1 > last1) ? base1 : last1);
    const uint32_t gemm1UsedCores = MLA_G * nSplit;

    // ---- 6. GEMM2 分核（只切 N，全部 aicNum 核参与）----
    const uint32_t n2Units = MLA_N2 / MLA_ALIGN16; // 320
    const uint32_t gemm2Cores = aicNum;
    const uint32_t base2 = n2Units / gemm2Cores;
    const uint32_t rem2 = n2Units % gemm2Cores;
    const uint32_t gemm2SingleNMax = MLA_ALIGN16 * (base2 + ((rem2 != 0U) ? 1U : 0U));
    const uint32_t gemm2UsedCores = gemm2Cores;
    if (base2 == 0U) {
        return ReportTilingFailure(context, "GEMM2 cores " + std::to_string(gemm2Cores) +
                                                 " > N units " + std::to_string(n2Units));
    }

    // GEMM2 的 B 按转置使用（存储 [N2, K2]），每个 K 行的连续长度只有 baseK × 2 B。
    // r2 优化（MDL 大块搬运）：baseN 取满单核 N 块（24 核 → 224），使每核 N 只落在 1 个
    // baseN tile 内 —— MDL 下每个 N tile 都要把 A 片沿 K 重扫一遍，N tile 数 2→1 直接把
    // GEMM2 的 A（c 片）GM→L1 请求量减半，MTE2 指令数随之减半。
    // baseK 取 64：库在保持 L0B 双缓冲（224×64×2×2 B = 57,344 ≤ 64 KB）下能容纳的最大值；
    // 请求 baseK=128 时 L0B 双缓冲需 114,688 > 64 KB，库会静默回落单缓冲/降 baseK，故显式取 64。
    // L0C：baseM × baseN × 4 B = 96×224×4 = 86,016 ≤ 128 KB。
    const uint32_t gemm2BaseK = 64U;
    uint32_t gemm2BaseN = (gemm2SingleNMax / MLA_ALIGN16) * MLA_ALIGN16;
    // L0B 双缓冲上限：baseN × baseK × 2 B × 2 ≤ 64 KB → baseK=64 时 baseN ≤ 256
    if (gemm2BaseN > 256U) {
        gemm2BaseN = 256U;
    }
    if (gemm2BaseN < 4U * MLA_ALIGN16) {
        gemm2BaseN = 4U * MLA_ALIGN16;
    }

    // ---- 7. 两套 TCubeTiling（单核 MatmulApiTiling + Host 手工分核）----
    matmul_tiling::MatmulApiTiling mm1Tiling(platform);
    const int32_t mm1SetATypeRc = mm1Tiling.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BFLOAT16, false);
    if (mm1SetATypeRc != 0) {
        return ReportTilingFailure(context, "GEMM1 SetAType failed, rc=" + std::to_string(mm1SetATypeRc));
    }
    const int32_t mm1SetBTypeRc = mm1Tiling.SetBType(
        matmul_tiling::TPosition::GM,
        nzConv ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BFLOAT16, false); // GEMM1：B 非转置（nzConv 时为 NZ 副本）
    if (mm1SetBTypeRc != 0) {
        return ReportTilingFailure(context, "GEMM1 SetBType failed, rc=" + std::to_string(mm1SetBTypeRc));
    }
    // c 内部布局改为 NZ：GEMM1 直接写 NZ，GEMM2 直接读 NZ，省掉 GEMM2 A 的 ND→NZ。
    const int32_t mm1SetCTypeRc = mm1Tiling.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ,
        matmul_tiling::DataType::DT_BFLOAT16);
    if (mm1SetCTypeRc != 0) {
        return ReportTilingFailure(context, "GEMM1 SetCType failed, rc=" + std::to_string(mm1SetCTypeRc));
    }
    const int32_t mm1SetShapeRc = mm1Tiling.SetShape(
        static_cast<int32_t>(pipeTileRows), static_cast<int32_t>(gemm1SingleNMax),
        static_cast<int32_t>(MLA_K1));
    if (mm1SetShapeRc != 0) {
        return ReportTilingFailure(context, "GEMM1 SetShape failed, rc=" + std::to_string(mm1SetShapeRc) +
                                                 " (M=" + std::to_string(pipeTileRows) +
                                                 " N=" + std::to_string(gemm1SingleNMax) +
                                                 " K=" + std::to_string(MLA_K1) + ")");
    }
    // r2 优化（MDL 大块搬运）：GEMM1 不调用 SetFixSplit，baseN/baseK 交回库选择
    // （库默认 baseN=256 / baseK=64，L0B 双缓冲 256×64×2×2 B = 65,536 = 64 KB 恰好占满）。
    // 实测（S=96，MDL 调度）：库默认时 GEMM1 阶段 102.3 us；强制 baseN=320 会被库降 baseK=32
    // （L0B 双缓冲放不下 320×64×2×2 B），阶段反而 116.4 us。故显式保持库默认。
    // Host 仅 3 参：C 的 8192 行跨度由 Kernel 侧 SetOrgShape 的 orgKc 承担（P4 首个样例核对）
    const int32_t mm1OrgShapeRc =
        mm1Tiling.SetOrgShape(static_cast<int32_t>(paddedM), static_cast<int32_t>(MLA_N1),
                              static_cast<int32_t>(MLA_K1));
    if (mm1OrgShapeRc != 0) {
        return ReportTilingFailure(context, "GEMM1 SetOrgShape failed, rc=" + std::to_string(mm1OrgShapeRc) +
                                                 " (M=" + std::to_string(paddedM) +
                                                 " N=" + std::to_string(MLA_N1) +
                                                 " K=" + std::to_string(MLA_K1) + ")");
    }
    const int64_t mm1GetTilingRc = mm1Tiling.GetTiling(tiling->gemm1Tiling);
    if (mm1GetTilingRc != 0) {
        return ReportTilingFailure(context, "GEMM1 GetTiling failed, rc=" + std::to_string(mm1GetTilingRc) +
                                                 " (M=" + std::to_string(pipeTileRows) +
                                                 " N=" + std::to_string(gemm1SingleNMax) +
                                                 " K=" + std::to_string(MLA_K1) + ")");
    }

    matmul_tiling::MatmulApiTiling mm2Tiling(platform);
    // c 是 GEMM1 的 NZ 输出，GEMM2 A 直接按 NZ 读取。
    const int32_t mm2SetATypeRc = mm2Tiling.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ,
        matmul_tiling::DataType::DT_BFLOAT16, false);
    if (mm2SetATypeRc != 0) {
        return ReportTilingFailure(context, "GEMM2 SetAType failed, rc=" + std::to_string(mm2SetATypeRc));
    }
    const int32_t mm2SetBTypeRc = mm2Tiling.SetBType(
        matmul_tiling::TPosition::GM,
        nzConv ? matmul_tiling::CubeFormat::NZ : matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BFLOAT16, true); // GEMM2：B 转置（nzConv 时为 NZ 副本）
    if (mm2SetBTypeRc != 0) {
        return ReportTilingFailure(context, "GEMM2 SetBType failed, rc=" + std::to_string(mm2SetBTypeRc));
    }
    const int32_t mm2SetCTypeRc = mm2Tiling.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BFLOAT16);
    if (mm2SetCTypeRc != 0) {
        return ReportTilingFailure(context, "GEMM2 SetCType failed, rc=" + std::to_string(mm2SetCTypeRc));
    }
    const int32_t mm2SetShapeRc = mm2Tiling.SetShape(
        static_cast<int32_t>(totalS), static_cast<int32_t>(gemm2SingleNMax),
        static_cast<int32_t>(MLA_K2));
    if (mm2SetShapeRc != 0) {
        return ReportTilingFailure(context, "GEMM2 SetShape failed, rc=" + std::to_string(mm2SetShapeRc) +
                                                 " (M=" + std::to_string(totalS) +
                                                 " N=" + std::to_string(gemm2SingleNMax) +
                                                 " K=" + std::to_string(MLA_K2) + ")");
    }
    const int32_t mm2OrgShapeRc =
        mm2Tiling.SetOrgShape(static_cast<int32_t>(totalS), static_cast<int32_t>(MLA_N2),
                              static_cast<int32_t>(MLA_K2));
    if (mm2OrgShapeRc != 0) {
        return ReportTilingFailure(context, "GEMM2 SetOrgShape failed, rc=" + std::to_string(mm2OrgShapeRc) +
                                                 " (M=" + std::to_string(totalS) +
                                                 " N=" + std::to_string(MLA_N2) +
                                                 " K=" + std::to_string(MLA_K2) + ")");
    }
    // baseM 交给库按 L0A/L0C 约束自选（-1 = 不指定），只锁定 baseN/baseK（见 §6 注释）
    const int32_t mm2FixSplitRc =
        mm2Tiling.SetFixSplit(-1, static_cast<int32_t>(gemm2BaseN), static_cast<int32_t>(gemm2BaseK));
    if (mm2FixSplitRc != 0) {
        return ReportTilingFailure(context, "GEMM2 SetFixSplit failed, rc=" + std::to_string(mm2FixSplitRc) +
                                                 " (baseM=-1 baseN=" + std::to_string(gemm2BaseN) +
                                                 " baseK=" + std::to_string(gemm2BaseK) + ")");
    }
    const int64_t mm2GetTilingRc = mm2Tiling.GetTiling(tiling->gemm2Tiling);
    if (mm2GetTilingRc != 0) {
        return ReportTilingFailure(context, "GEMM2 GetTiling failed, rc=" + std::to_string(mm2GetTilingRc) +
                                                 " (M=" + std::to_string(totalS) +
                                                 " N=" + std::to_string(gemm2SingleNMax) +
                                                 " K=" + std::to_string(MLA_K2) + ")");
    }

    // GetTiling 后按手工分核覆写 M/N/Ka/Kb、singleCoreM/singleCoreN；
    // usedCoreNum 保持单核 GetTiling 给出的值（P7：不改写为 (M/singleCoreM)*(N/singleCoreN)）
    tiling->gemm1Tiling.M = static_cast<int32_t>(pipeTileRows);
    tiling->gemm1Tiling.N = static_cast<int32_t>(gemm1SingleNMax);
    tiling->gemm1Tiling.Ka = static_cast<int32_t>(MLA_K1);
    tiling->gemm1Tiling.Kb = static_cast<int32_t>(MLA_K1);
    tiling->gemm1Tiling.singleCoreM = static_cast<int32_t>(pipeTileRows);
    tiling->gemm1Tiling.singleCoreN = static_cast<int32_t>(gemm1SingleNMax);
    tiling->gemm1Tiling.singleCoreK = static_cast<int32_t>(MLA_K1);

    tiling->gemm2Tiling.M = static_cast<int32_t>(totalS);
    tiling->gemm2Tiling.N = static_cast<int32_t>(gemm2SingleNMax);
    tiling->gemm2Tiling.Ka = static_cast<int32_t>(MLA_K2);
    tiling->gemm2Tiling.Kb = static_cast<int32_t>(MLA_K2);
    tiling->gemm2Tiling.singleCoreM = static_cast<int32_t>(totalS);
    tiling->gemm2Tiling.singleCoreN = static_cast<int32_t>(gemm2SingleNMax);
    tiling->gemm2Tiling.singleCoreK = static_cast<int32_t>(MLA_K2);

    // ---- 8. Matmul 片上临时空间实测核算（M-10 口径；AIC/AIV 独立、不叠加不互扣）----
    uint64_t l1Size = 0ULL;
    uint64_t l0cSize = 0ULL;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    if (l1Size == 0ULL) {
        return ReportTilingFailure(context, "GetCoreMemSize(L1) returned 0");
    }
    if (l0cSize == 0ULL) {
        return ReportTilingFailure(context, "GetCoreMemSize(L0_C) returned 0");
    }

    // ---- 8a. MDL 大块搬运调度参数覆写（性能优化 r2；不满足整除/L1 约束时自动保留库值）----
    const bool g1StepOk = ApplyMdlStepTiling(tiling->gemm1Tiling, 8, 2, l1Size);
    const bool g2StepOk = ApplyMdlStepTiling(tiling->gemm2Tiling, 8, 4, l1Size);

    matmul_tiling::SysTilingTempBufSize mm1Buf;
    matmul_tiling::SysTilingTempBufSize mm2Buf;
    const int32_t mm1BufRc = MatmulGetTmpBufSizeV2(tiling->gemm1Tiling, mm1Buf);
    if (mm1BufRc != 0) {
        return ReportTilingFailure(context, "MatmulGetTmpBufSizeV2(GEMM1) failed, rc=" + std::to_string(mm1BufRc) +
                                                 " (M=" + std::to_string(tiling->gemm1Tiling.M) +
                                                 " N=" + std::to_string(tiling->gemm1Tiling.N) +
                                                 " K=" + std::to_string(tiling->gemm1Tiling.Ka) +
                                                 " baseN=" + std::to_string(tiling->gemm1Tiling.baseN) +
                                                 " baseK=" + std::to_string(tiling->gemm1Tiling.baseK) + ")");
    }
    const int32_t mm2BufRc = MatmulGetTmpBufSizeV2(tiling->gemm2Tiling, mm2Buf);
    if (mm2BufRc != 0) {
        return ReportTilingFailure(context, "MatmulGetTmpBufSizeV2(GEMM2) failed, rc=" + std::to_string(mm2BufRc) +
                                                 " (M=" + std::to_string(tiling->gemm2Tiling.M) +
                                                 " N=" + std::to_string(tiling->gemm2Tiling.N) +
                                                 " K=" + std::to_string(tiling->gemm2Tiling.Ka) +
                                                 " baseN=" + std::to_string(tiling->gemm2Tiling.baseN) +
                                                 " baseK=" + std::to_string(tiling->gemm2Tiling.baseK) + ")");
    }
    if (static_cast<uint64_t>(mm1Buf.l1Size) > l1Size || static_cast<uint64_t>(mm2Buf.l1Size) > l1Size) {
        return ReportTilingFailure(context, "Matmul L1 need g1=" + std::to_string(mm1Buf.l1Size) +
                                                 " g2=" + std::to_string(mm2Buf.l1Size) +
                                                 " > L1 " + std::to_string(l1Size));
    }
    if (static_cast<uint64_t>(mm1Buf.l0cSize) > l0cSize || static_cast<uint64_t>(mm2Buf.l0cSize) > l0cSize) {
        return ReportTilingFailure(context, "Matmul L0C need g1=" + std::to_string(mm1Buf.l0cSize) +
                                                 " g2=" + std::to_string(mm2Buf.l0cSize) +
                                                 " > L0C " + std::to_string(l0cSize));
    }
    if (static_cast<uint64_t>(mm1Buf.ubSize) > usableUB || static_cast<uint64_t>(mm2Buf.ubSize) > usableUB) {
        return ReportTilingFailure(context, "Matmul UB peak g1=" + std::to_string(mm1Buf.ubSize) +
                                                 " g2=" + std::to_string(mm2Buf.ubSize) +
                                                 " > usableUB " + std::to_string(usableUB));
    }

    // ---- 9. workspace（用户区 r/c + 系统区）----
    const uint64_t rBytes =
        static_cast<uint64_t>(MLA_G) * static_cast<uint64_t>(paddedM) * static_cast<uint64_t>(MLA_K1) * 2ULL;
    const uint64_t cBytes =
        static_cast<uint64_t>(paddedM) * static_cast<uint64_t>(MLA_G) * static_cast<uint64_t>(MLA_N1) * 2ULL;
    const uint64_t cOffset = AlignUpU64(rBytes, WS_ALIGN);
    // nzConv 时追加 w1/w2 的 NZ 副本（各与 ND 原权重同字节数）
    const uint64_t w1NzBytes =
        nzConv ? static_cast<uint64_t>(MLA_G) * static_cast<uint64_t>(MLA_K1) * static_cast<uint64_t>(MLA_N1) * 2ULL
               : 0ULL;
    const uint64_t w2NzBytes =
        nzConv ? static_cast<uint64_t>(MLA_K2) * static_cast<uint64_t>(MLA_N2) * 2ULL : 0ULL;
    const uint64_t w1NzOffset = cOffset + AlignUpU64(cBytes, WS_ALIGN);
    const uint64_t w2NzOffset = w1NzOffset + AlignUpU64(w1NzBytes, WS_ALIGN);
    const uint64_t userWorkspaceSize = w2NzOffset + AlignUpU64(w2NzBytes, WS_ALIGN);
    const uint32_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    if (sysWorkspaceSize == UINT32_MAX) {
        return ReportTilingFailure(context, "GetLibApiWorkSpaceSize() returned UINT32_MAX");
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(userWorkspaceSize + static_cast<uint64_t>(sysWorkspaceSize));

    // ---- 10. TilingData 字段落盘 ----
    tiling->totalS = totalS;
    tiling->paddedM = paddedM;
    tiling->padRows = padRows;
    tiling->tileS = tileS;
    tiling->pipeTileNum = pipeTileNum;
    tiling->pipeTileRows = pipeTileRows;
    tiling->sTileNum = sTileNum;
    tiling->ropeTaskNum = ropeTaskNum;
    tiling->ropeTasksPerCore = ropeTasksPerCore;
    tiling->aivUsedCores = aivUsedCores;
    tiling->aicUsedCores = aicNum;

    tiling->gemm1CoresPerBatch = nSplit;
    tiling->gemm1NUnitsBase = base1;
    tiling->gemm1NUnitsLast = last1;
    tiling->gemm1SingleNMax = gemm1SingleNMax;
    tiling->gemm1SingleM = pipeTileRows;
    tiling->gemm1UsedCores = gemm1UsedCores;

    tiling->gemm2NUnitsBase = base2;
    tiling->gemm2NUnitsRem = rem2;
    tiling->gemm2SingleNMax = gemm2SingleNMax;
    tiling->gemm2SingleM = totalS;
    tiling->gemm2UsedCores = gemm2UsedCores;

    tiling->sysWorkspaceSize = sysWorkspaceSize;
    tiling->rOffset = 0ULL;
    tiling->cOffset = cOffset;
    tiling->userWorkspaceSize = userWorkspaceSize;
    tiling->nzConv = nzConv ? 1U : 0U;
    tiling->nzPad = 0U;
    tiling->w1NzOffset = w1NzOffset;
    tiling->w2NzOffset = w2NzOffset;

    // ---- 11. P1/P2/P4/P6/P7 取证日志（首个样例据此核对）----
    OP_LOGI(context,
            "tiling: S=%u paddedM=%u padRows=%u aicNum=%u aivNum=%u ubSize=%lu usableUB=%lu "
            "sysWs=%u userWs=%lu totalWs=%lu nzConv=%u",
            totalS, paddedM, padRows, aicNum, aivNum, static_cast<unsigned long>(ubSize),
            static_cast<unsigned long>(usableUB), sysWorkspaceSize, static_cast<unsigned long>(userWorkspaceSize),
            static_cast<unsigned long>(userWorkspaceSize + static_cast<uint64_t>(sysWorkspaceSize)),
            static_cast<unsigned>(nzConv ? 1U : 0U));
    OP_LOGI(context,
            "RoPE: tileS=%u pipeTileNum=%u pipeTileRows=%u sTileNum=%u ropeTaskNum=%u "
            "ropeTasksPerCore=%u aivUsedCores=%u ropeUB=%lu (fixedUB=%lu rowUB=%lu)",
            tileS, pipeTileNum, pipeTileRows, sTileNum, ropeTaskNum, ropeTasksPerCore, aivUsedCores,
            static_cast<unsigned long>(ropeUB), static_cast<unsigned long>(ROPE_FIXED_UB),
            static_cast<unsigned long>(ROPE_ROW_UB));
    OP_LOGI(context,
            "GEMM1[%s]: nSplit=%u base=%u last=%u singleNMax=%u singleM=%u usedCores=%u | tiling "
            "usedCoreNum=%d M=%d N=%d Ka=%d Kb=%d singleCoreM=%d singleCoreN=%d singleCoreK=%d "
            "baseM=%d baseN=%d baseK=%d stepM=%d stepN=%d stepKa=%d stepKb=%d depthA1=%d depthB1=%d "
            "dbL0A=%d dbL0B=%d dbL0C=%d",
            (g1StepOk ? "r2step" : "libstep"),
            nSplit, base1, last1, gemm1SingleNMax, pipeTileRows, gemm1UsedCores, tiling->gemm1Tiling.usedCoreNum,
            tiling->gemm1Tiling.M, tiling->gemm1Tiling.N, tiling->gemm1Tiling.Ka, tiling->gemm1Tiling.Kb,
            tiling->gemm1Tiling.singleCoreM, tiling->gemm1Tiling.singleCoreN, tiling->gemm1Tiling.singleCoreK,
            tiling->gemm1Tiling.baseM, tiling->gemm1Tiling.baseN, tiling->gemm1Tiling.baseK,
            tiling->gemm1Tiling.stepM, tiling->gemm1Tiling.stepN, tiling->gemm1Tiling.stepKa,
            tiling->gemm1Tiling.stepKb, tiling->gemm1Tiling.depthA1, tiling->gemm1Tiling.depthB1,
            tiling->gemm1Tiling.dbL0A, tiling->gemm1Tiling.dbL0B, tiling->gemm1Tiling.dbL0C);
    OP_LOGI(context,
            "GEMM2[%s]: base=%u rem=%u singleNMax=%u fixBaseN=%u fixBaseK=%u singleM=%u usedCores=%u | tiling usedCoreNum=%d "
            "M=%d N=%d Ka=%d Kb=%d singleCoreM=%d singleCoreN=%d singleCoreK=%d baseM=%d baseN=%d baseK=%d "
            "stepM=%d stepN=%d stepKa=%d stepKb=%d depthA1=%d depthB1=%d dbL0A=%d dbL0B=%d dbL0C=%d "
            "(host SetBType isTrans=true)",
            (g2StepOk ? "r2step" : "libstep"),
            base2, rem2, gemm2SingleNMax, gemm2BaseN, gemm2BaseK, totalS, gemm2UsedCores, tiling->gemm2Tiling.usedCoreNum,
            tiling->gemm2Tiling.M, tiling->gemm2Tiling.N, tiling->gemm2Tiling.Ka, tiling->gemm2Tiling.Kb,
            tiling->gemm2Tiling.singleCoreM, tiling->gemm2Tiling.singleCoreN, tiling->gemm2Tiling.singleCoreK,
            tiling->gemm2Tiling.baseM, tiling->gemm2Tiling.baseN, tiling->gemm2Tiling.baseK,
            tiling->gemm2Tiling.stepM, tiling->gemm2Tiling.stepN, tiling->gemm2Tiling.stepKa,
            tiling->gemm2Tiling.stepKb, tiling->gemm2Tiling.depthA1, tiling->gemm2Tiling.depthB1,
            tiling->gemm2Tiling.dbL0A, tiling->gemm2Tiling.dbL0B, tiling->gemm2Tiling.dbL0C);
    OP_LOGI(context,
            "MatmulGetTmpBufSize: g1 ub=%d l1=%d l0c=%d | g2 ub=%d l1=%d l0c=%d | capacity "
            "l1=%lu l0c=%lu usableUB=%lu",
            mm1Buf.ubSize, mm1Buf.l1Size, mm1Buf.l0cSize, mm2Buf.ubSize, mm2Buf.l1Size, mm2Buf.l0cSize,
            static_cast<unsigned long>(l1Size), static_cast<unsigned long>(l0cSize),
            static_cast<unsigned long>(usableUB));

    // ---- 12. TilingKey / batchmode / blockDim ----
    ASCENDC_TPL_SEL_PARAM(context, 0U);
    const ge::graphStatus scheduleModeRc = context->SetScheduleMode(1);
    if (scheduleModeRc != ge::GRAPH_SUCCESS) {
        return ReportTilingFailure(context, "SetScheduleMode(1) failed, rc=" +
                                                 std::to_string(static_cast<unsigned int>(scheduleModeRc)));
    }
    const ge::graphStatus blockDimRc = context->SetBlockDim(aicNum);
    if (blockDimRc != ge::GRAPH_SUCCESS) {
        return ReportTilingFailure(context, "SetBlockDim(" + std::to_string(aicNum) + ") failed, rc=" +
                                                 std::to_string(static_cast<unsigned int>(blockDimRc)));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMlaEpilogue([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MlaEpilogueCompileInfo {}; // 必须定义，入图场景依赖

IMPL_OP_OPTILING(FiMlaEpilogue)
    .Tiling(TilingMlaEpilogue)
    .TilingParse<MlaEpilogueCompileInfo>(TilingParseForMlaEpilogue);

} // namespace optiling
