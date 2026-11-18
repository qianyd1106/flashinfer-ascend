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
 * \file mla_epilogue_gemm.h
 * \brief mla_epilogue AIC 侧计算上下文：GEMM1 / GEMM2（DESIGN §3.5）
 *
 * 约束（DESIGN §3.5 / DC-2 / DC-5 / DC-8）：
 *  - 仅在 ASCEND_IS_AIC 分支内初始化与调用；本文件不含任何 SyncAll，屏障只存在于 Kernel 入口公共控制流；
 *  - GEMM1 B 使用 bType（ISTRANS=false），运行时 SetTensorB(..., false)；
 *    GEMM2 B 使用 bTypeTrans（ISTRANS=true），运行时 SetTensorB(..., true)，三处（Host SetBType /
 *    MatmulType ISTRANS / 运行时 isTransposeB）必须一致（P6）；
 *  - 手工分核：每核自行用核号推导 (g, nIdx) 与 n0/curN，再以 SetSingleShape + 预偏移指针下发；
 *    不依赖 TCubeTiling.usedCoreNum 的均匀切分（不改写 usedCoreNum，P7）；
 *    GEMM1 的 N 切分按 128 B（64 元素）对齐，消除 MTE2 非对齐请求的 sector 浪费；
 *  - REGIST_MATMUL_OBJ 必须早于本 pipe 的任何 InitBuffer（由 Kernel 入口保证调用顺序）。
 */

#ifndef __MLA_EPILOGUE_GEMM_H__
#define __MLA_EPILOGUE_GEMM_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "mla_epilogue_tiling_data.h"

namespace NsMlaEpilogue {
using namespace AscendC;

// Matmul 矩阵类型（DESIGN §3.5）：GEMM1 B 非转置；GEMM2 B 转置，类型必须带 ISTRANS=true
using aType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using bType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;         // ISTRANS=false：GEMM1 非转置
using bTypeTrans = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>; // ISTRANS=true：GEMM2 B 转置
using cType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using biasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

/**
 * @brief AIC 侧计算上下文：持有 mm1 / mm2 与 GM 视图
 *
 * 构造只声明 Matmul 对象，不调用任何 Matmul API；Init/Gemm1Tile/Gemm2 全部只在 AIC 分支内执行。
 */
class AicContext {
public:
    __aicore__ inline void Init(const MlaEpilogueTilingData& td, TPipe* pipe, GM_ADDR userWs, GM_ADDR w1,
                                GM_ADDR w2, GM_ADDR y)
    {
        // 用户 workspace 视图：r[8, paddedM, 4096] / c[paddedM, 8192]（DESIGN §3.3 / §3.9）
        rGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWs + td.rOffset));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWs + td.cOffset));
        w1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w1));
        w2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w2));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));

        // 系统 workspace 交给框架；必须在任何 InitBuffer 之前（DESIGN §3.4 第 7 步）
        REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), mm1_, &td.gemm1Tiling, mm2_, &td.gemm2Tiling);
    }

    /**
     * @brief GEMM1 第 t 个流水段：r[g] 的第 [m0, m0+curM) 行 × w1[g] 的本核 N 列块 → c 对应块
     *        纯计算体，不含 SyncAll（屏障由入口公共控制流给出）。
     */
    __aicore__ inline void Gemm1Tile(uint32_t t, const MlaEpilogueTilingData& td)
    {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        if (coreIdx >= td.gemm1UsedCores) {
            return; // 未承担本段计算的核：只跳过计算体，仍参与入口屏障
        }
        const uint32_t nSplit = td.gemm1CoresPerBatch;
        const uint32_t g = coreIdx / nSplit;
        const uint32_t nIdx = coreIdx - g * nSplit;
        // 128 B 对齐分核（性能优化）：前 nSplit-1 个核各取 gemm1NUnitsBase 个 16 元素单元，
        // 最后一个核取 gemm1NUnitsLast 个（吃掉余数）。Host 保证 base 是 4 单元（64 元素 = 128 B）
        // 的整数倍，因此每个核的列块起点 n0 都落在 128 B 边界上，MTE2 的 GM→L1 请求不跨 sector。
        // 约束校验（base*(nSplit-1) + last == n1Units）在 Host tiling 完成。
        const bool isLast = ((nIdx + 1U) >= nSplit);
        const uint32_t nUnits = isLast ? td.gemm1NUnitsLast : td.gemm1NUnitsBase;
        const uint32_t n0 = MLA_ALIGN16 * (isLast ? ((nSplit - 1U) * td.gemm1NUnitsBase) :
                                                     (nIdx * td.gemm1NUnitsBase));
        const uint32_t curN = MLA_ALIGN16 * nUnits;

        const uint32_t m0 = t * td.pipeTileRows;
        const uint32_t restM = td.paddedM - m0;
        const uint32_t curM = (td.pipeTileRows < restM) ? td.pipeTileRows : restM;

        // 5 参 SetOrgShape：orgKc = MLA_G * N1 = 8192 为 C（c 的 [paddedM, 8192]）行跨度，
        // 与 B 的 orgN（N1 = 1024）不同；必须在 SetTensorA/SetTensorB/SetSingleShape 之前调用（P4）
        mm1_.SetOrgShape(static_cast<int32_t>(td.paddedM), static_cast<int32_t>(MLA_N1), static_cast<int32_t>(MLA_K1),
                         static_cast<int32_t>(MLA_K1), static_cast<int32_t>(MLA_G * MLA_N1));
        // r 布局 [8, paddedM, 4096]，第 t 段从行 m0 起，行跨度仍为 K1
        mm1_.SetTensorA(rGm_[static_cast<uint64_t>(g) * td.paddedM * MLA_K1 + static_cast<uint64_t>(m0) * MLA_K1],
                        false);
        // w1[g] 为 [K1, N1]，本核列块起点 n0；bType 为 ISTRANS=false → 运行期只能传 false
        mm1_.SetTensorB(w1Gm_[static_cast<uint64_t>(g) * MLA_K1 * MLA_N1 + n0], false);
        // 覆盖内建单核形状：M 为本段行数（末段可小），N 为本核列块（320/320/384，128 B 对齐）
        mm1_.SetSingleShape(static_cast<int32_t>(curM), static_cast<int32_t>(curN), static_cast<int32_t>(MLA_K1));
        // c 布局 [paddedM, 8192]（行内 [g, m] 排布）；每行只写本核的 N 列块
        mm1_.IterateAll(cGm_[static_cast<uint64_t>(m0) * (MLA_G * MLA_N1) + static_cast<uint64_t>(g) * MLA_N1 + n0],
                        0, false);
    }

    /**
     * @brief GEMM2：c_flat[S, 8192] @ w2[N2, K2]^T → y[S, 5120]（本核 N 列块）
     *        纯计算体，不含 SyncAll。
     */
    __aicore__ inline void Gemm2(const MlaEpilogueTilingData& td)
    {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        if (coreIdx >= td.gemm2UsedCores) {
            return;
        }
        const uint32_t hasRem = (coreIdx < td.gemm2NUnitsRem) ? 1U : 0U;
        const uint32_t nUnits = td.gemm2NUnitsBase + hasRem;
        const uint32_t n0 = MLA_ALIGN16 * (coreIdx * td.gemm2NUnitsBase +
                                           ((coreIdx < td.gemm2NUnitsRem) ? coreIdx : td.gemm2NUnitsRem));
        const uint32_t curN = MLA_ALIGN16 * nUnits;

        // 3 参重载：orgKa = orgKb = orgKc = K2；B 的 N 与 C 的 N 同为 N2 = 5120
        mm2_.SetOrgShape(static_cast<int32_t>(td.totalS), static_cast<int32_t>(MLA_N2), static_cast<int32_t>(MLA_K2));
        mm2_.SetTensorA(cGm_[0], false); // c 即 c_flat[S, 8192]，K 维连续
        // w2 按 B 转置使用（存储 [N2, K2]），bTypeTrans 为 ISTRANS=true → 运行期传 true（P6）
        mm2_.SetTensorB(w2Gm_[static_cast<uint64_t>(n0) * MLA_K2], true);
        mm2_.SetSingleShape(static_cast<int32_t>(td.totalS), static_cast<int32_t>(curN),
                            static_cast<int32_t>(MLA_K2));
        mm2_.IterateAll(yGm_[n0], 0, false); // 直接写 y[S,5120]，消除 TensorMove
    }

    /**
     * @brief GEMM1 全部 M-段下发完毕后关闭 mm1_（必须在 Gemm2 之前调用）
     *
     * 根因（issue_20261111_matmul-end-mlayout_01）：mm1_ 与 mm2_ 共享同一个 TPipe。
     * 若 mm1_ 不 End()，其 CubeInBuffer 占用的 L1 队列事件（TQueBind eventId）与 L0 池缓存不会释放；
     * mm2_ 在推进第 2 个及以上 baseM 块（CopyCubeInA->Reset → qid_.FreeBuffer）时会死锁在
     * 单元 flag 的硬件等待上（AICore 100%、无异常、无 stranded cqe 的挂死）。
     * GEMM2 只有在 mm2_ 的 M 方向切成 >= 2 个 baseM 块时才会触发该路径（S 较小时单块不触发）。
     *
     * End() 会把硬件控制位 MM_LAYOUT_MODE（C 为 ND 时为 1，由 SchedulerBase::Init 在
     * REGIST_MATMUL_OBJ 阶段对两个对象各设置一次）清零，而 mm2_ 的 Init 不会再执行，
     * 故必须用 SetMMRowMajor() 恢复，否则 GEMM2 的 C 输出布局错误。
     */
    __aicore__ inline void CloseGemm1()
    {
        mm1_.End();
        AscendC::SetMMRowMajor();
    }

private:
    // 性能优化（r2）：Matmul 调度由 Norm 切换为 MDL —— MDL 按 TCubeTiling 的
    // stepKa/stepKb/depthA1/depthB1 以大块 L1 tile 搬运（A: baseM × stepKa×baseK，
    // B: baseN × stepKb×baseK），把 GM→L1 的 MTE2 指令数降为约 1/4~1/8，
    // 单条搬运粒度从 ~16/25 KiB 提升到 ~80~90 KiB（对齐库 kernel 的 51~110 KiB）。
    // baseM/baseN/baseK 不变 → L0 累加顺序不变，结果与 Norm 逐位一致。
    Matmul<aType, bType, cType, biasType, CFG_MDL> mm1_;      // GEMM1：B 非转置

    Matmul<aType, bTypeTrans, cType, biasType, CFG_MDL> mm2_; // GEMM2：B 转置

    GlobalTensor<bfloat16_t> rGm_;  // [8, paddedM, 4096]
    GlobalTensor<bfloat16_t> cGm_;  // [paddedM, 8192]
    GlobalTensor<bfloat16_t> w1Gm_; // [8, 4096, 1024]
    GlobalTensor<bfloat16_t> w2Gm_; // [5120, 8192]
    GlobalTensor<bfloat16_t> yGm_;  // [S, 5120]
};

} // namespace NsMlaEpilogue

#endif // __MLA_EPILOGUE_GEMM_H__
