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
 * \file mla_epilogue.cpp
 * \brief mla_epilogue Kernel 入口（arch22 / Ascend910_93）
 *
 * 单 MIX Kernel（AIC:AIV = 1:2，由 ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2) 声明）：
 *   AIV：RoPE（x 的每 head [448,512) 共 64 维 rotate_every_two）→ r
 *   AIC：GEMM1（r[g] × w1[g] → c）分段流水，GEMM2（c_flat × w2^T → y）
 *
 * 同步契约（DC-3 / §3.5）：
 *   - 屏障序列只存在于本文件的公共控制流，AicContext / RopeContext 内部不含任何 SyncAll；
 *   - 所有启动核（24 AIC + 48 AIV）都执行每一次 SyncAll<false>()；
 *     未承担本段计算任务的核只跳过计算循环体，入口不设任何早退 return。
 *
 * 分发契约（DC-1 / DC-4）：
 *   - 单 TilingKey（SCH_MODE = 0），编译期模板选择；不使用 TILING_KEY_IS 分支。
 *   - REGISTER_TILING_DEFAULT 在 Kernel 入口、GET_TILING_DATA_WITH_STRUCT 之前注册。
 */

#define ASCENDC_CUBE_ONLY // Matmul 仅在 AIC 侧调用（DESIGN §3.1/§3.5）；必须先于 matmul_intf.h

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"

#include "arch22/mla_epilogue_tiling_data.h"
#include "arch22/mla_epilogue_tiling_key.h"
#include "arch22/mla_epilogue_gemm.h"
#include "arch22/mla_epilogue_rope.h"
#include "arch22/mla_epilogue_nzconv.h"

using namespace AscendC;

/**
 * @brief mla_epilogue 全功能 Kernel 入口（单 TilingKey）
 *
 * @param x         [S, 1, 64, 512] bf16，行内布局 [8 g][8 h][512 d]
 * @param cos       [S, 1, 1, 64] fp32
 * @param sin       [S, 1, 1, 64] fp32
 * @param w1        [8, 4096, 1024] bf16
 * @param w2        [5120, 8192] bf16
 * @param y         [S, 5120] bf16（直接写用户输出）
 * @param workspace 系统 workspace + 用户 workspace（r / c 中间结果）
 * @param tiling    MlaEpilogueTilingData
 */
template <uint32_t SCH_MODE>
__global__ __aicore__ void fi_mla_epilogue(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR w1, GM_ADDR w2, GM_ADDR y,
                                        GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // AIC:AIV = 1:2，SyncAll<false> 硬同步前提

    if constexpr (SCH_MODE == 0U) {
        REGISTER_TILING_DEFAULT(MlaEpilogueTilingData); // 必须在 GET_TILING_DATA_WITH_STRUCT 之前
        GET_TILING_DATA_WITH_STRUCT(MlaEpilogueTilingData, tilingData, tiling);
        GM_ADDR userWs = GetUserWorkspace(workspace); // 系统 workspace 之外的用户区（§3.4 第 7 步）
        TPipe pipe;

        NsMlaEpilogue::AicContext aic;  // AIC 侧计算上下文：AIV 侧只构造、不调用
        NsMlaEpilogue::RopeContext rope; // AIV 侧计算上下文：AIC 侧只构造、不调用
        NsMlaEpilogue::NzConvContext nzconv; // AIV 侧：大 S 时把 w1/w2 预转成 NZ（nzConv=1）

        if ASCEND_IS_AIC {
            aic.Init(tilingData, &pipe, userWs, w1, w2, y); // REGIST_MATMUL_OBJ 早于本 pipe 的任何 InitBuffer
        }
        if ASCEND_IS_AIV {
            rope.Init(tilingData, &pipe, userWs, x, cos, sin);
            if (tilingData.nzConv != 0U) {
                nzconv.Init(tilingData, &pipe, userWs, w1, w2);
            }
        }

        // ===== 公共控制流：AIC/AIV 执行同一份屏障序列，计算体按 ASCEND_IS_AIC/AIV 隔离 =====
        if ASCEND_IS_AIV {
            rope.Tile(0U, tilingData); // 前导段：只做第 0 段 RoPE
            if (tilingData.nzConv != 0U) {
                nzconv.ConvW1(tilingData); // w1 NZ 必须在 GEMM1 之前就绪（与 RoPE 串行，同在 AIV）
            }
        }
        SyncAll<false>(); // B0：r 第 0 段（nzConv 时还有 w1 NZ）就绪

        for (uint32_t t = 0U; t < tilingData.pipeTileNum; ++t) {
            if ASCEND_IS_AIC {
                aic.Gemm1Tile(t, tilingData); // GEMM1 第 t 段（纯计算体，不含 SyncAll）
            }
            if ASCEND_IS_AIV {
                if ((t + 1U) < tilingData.pipeTileNum) {
                    rope.Tile(t + 1U, tilingData); // 与 Gemm1Tile(t) 并发：RoPE(t+1) ∥ GEMM1(t)
                } else if (tilingData.nzConv != 0U) {
                    nzconv.ConvW2(tilingData); // w2 NZ 只需在 GEMM2 前就绪：藏在 GEMM1 最后一段
                }
            }
            SyncAll<false>(); // B(t+1)：Gemm1Tile(t) 与 RopeTile(t+1)（或 w2 转换）均完成
        }

        if ASCEND_IS_AIC {
            aic.CloseGemm1();       // GEMM1 全部段已完成：End mm1_，释放其 L1 队列事件/L0 池缓存
            aic.Gemm2(tilingData);  // 最后一次 B 同时充当 GEMM1→GEMM2 屏障
        }
    }
}
