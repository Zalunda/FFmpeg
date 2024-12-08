/**
 * Copyright (c) 2016 Davinder Singh (DSM_) <ds.mudhar<@gmail.com>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVFILTER_MOTION_ESTIMATION_H
#define AVFILTER_MOTION_ESTIMATION_H

#include <stdint.h>

#define AV_ME_METHOD_ESA        1
#define AV_ME_METHOD_TSS        2
#define AV_ME_METHOD_TDLS       3
#define AV_ME_METHOD_NTSS       4
#define AV_ME_METHOD_FSS        5
#define AV_ME_METHOD_DS         6
#define AV_ME_METHOD_HEXBS      7
#define AV_ME_METHOD_EPZS       8
#define AV_ME_METHOD_UMH        9

typedef struct AVMotionEstPredictor2 {
    int mvs[10][2];
    int nb;
} AVMotionEstPredictor2;

typedef struct AVMotionEstContext2 {
    uint8_t *data_cur, *data_ref;
    int linesize;

    int mb_size;
    int search_param;

    int width;
    int height;

    int x_min;
    int x_max;
    int y_min;
    int y_max;

    int pred_x;     ///< median predictor x
    int pred_y;     ///< median predictor y
    AVMotionEstPredictor2 preds[2];

    uint64_t (*get_cost)(struct AVMotionEstContext2 *me_ctx, int x_mb, int y_mb,
                         int mv_x, int mv_y);
} AVMotionEstContext2;

void ff_me_init_context(AVMotionEstContext2 *me_ctx, int mb_size, int search_param,
                        int width, int height, int x_min, int x_max, int y_min, int y_max);

uint64_t ff_me_cmp_sad(AVMotionEstContext2 *me_ctx, int x_mb, int y_mb, int x_mv, int y_mv);

uint64_t ff_me_search_tdls2(AVMotionEstContext2 *me_ctx, int x_mb, int y_mb, int *mv);

#endif /* AVFILTER_MOTION_ESTIMATION_H */
