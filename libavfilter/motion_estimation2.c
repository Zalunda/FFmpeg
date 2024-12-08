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

#include "libavutil/common.h"
#include "motion_estimation2.h"

static const int8_t sqr1[8][2]  = {{ 0,-1}, { 0, 1}, {-1, 0}, { 1, 0}, {-1,-1}, {-1, 1}, { 1,-1}, { 1, 1}};
static const int8_t dia1[4][2]  = {{-1, 0}, { 0,-1}, { 1, 0}, { 0, 1}};
static const int8_t dia2[8][2]  = {{-2, 0}, {-1,-1}, { 0,-2}, { 1,-1}, { 2, 0}, { 1, 1}, { 0, 2}, {-1, 1}};
static const int8_t hex2[6][2]  = {{-2, 0}, {-1,-2}, {-1, 2}, { 1,-2}, { 1, 2}, { 2, 0}};
static const int8_t hex4[16][2] = {{-4,-2}, {-4,-1}, {-4, 0}, {-4, 1}, {-4, 2},
                                   { 4,-2}, { 4,-1}, { 4, 0}, { 4, 1}, { 4, 2},
                                   {-2, 3}, { 0, 4}, { 2, 3}, {-2,-3}, { 0,-4}, { 2,-3}};

#define COST_MV(x, y)\
do {\
    cost = me_ctx->get_cost(me_ctx, x_mb, y_mb, x, y);\
    if (cost < cost_min) {\
        cost_min = cost;\
        mv[0] = x;\
        mv[1] = y;\
    }\
} while(0)

#define COST_P_MV(x, y)\
if (x >= x_min && x <= x_max && y >= y_min && y <= y_max)\
    COST_MV(x, y);

void ff_me_init_context2(AVMotionEstContext2 *me_ctx, int mb_size, int search_param,
                        int width, int height, int x_min, int x_max, int y_min, int y_max)
{
    me_ctx->width = width;
    me_ctx->height = height;
    me_ctx->mb_size = mb_size;
    me_ctx->search_param = search_param;
    me_ctx->get_cost = &ff_me_cmp_sad;
    me_ctx->x_min = x_min;
    me_ctx->x_max = x_max;
    me_ctx->y_min = y_min;
    me_ctx->y_max = y_max;
}

uint64_t ff_me_cmp_sad2(AVMotionEstContext2 *me_ctx, int x_mb, int y_mb, int x_mv, int y_mv)
{
    const int linesize = me_ctx->linesize;
    uint8_t *data_ref = me_ctx->data_ref;
    uint8_t *data_cur = me_ctx->data_cur;
    uint64_t sad = 0;
    int i, j;

    data_ref += y_mv * linesize;
    data_cur += y_mb * linesize;

    for (j = 0; j < me_ctx->mb_size; j++)
        for (i = 0; i < me_ctx->mb_size; i++)
            sad += FFABS(data_ref[x_mv + i + j * linesize] - data_cur[x_mb + i + j * linesize]);

    return sad;
}

uint64_t ff_me_search_tdls2(AVMotionEstContext2 *me_ctx, int x_mb, int y_mb, int *mv)
{
    int x, y;
    int x_min = FFMAX(me_ctx->x_min, x_mb - me_ctx->search_param);
    int y_min = FFMAX(me_ctx->y_min, y_mb - me_ctx->search_param);
    int x_max = FFMIN(x_mb + me_ctx->search_param, me_ctx->x_max);
    int y_max = FFMIN(y_mb + me_ctx->search_param, me_ctx->y_max);
    uint64_t cost, cost_min;
    int step = ROUNDED_DIV(me_ctx->search_param, 2);
    int i;

    mv[0] = x_mb;
    mv[1] = y_mb;

    if (!(cost_min = me_ctx->get_cost(me_ctx, x_mb, y_mb, x_mb, y_mb)))
        return cost_min;

    do {
        x = mv[0];
        y = mv[1];

        for (i = 0; i < 4; i++)
            COST_P_MV(x + dia1[i][0] * step, y + dia1[i][1] * step);

        if (x == mv[0] && y == mv[1])
            step = step >> 1;

    } while (step > 0);

    return cost_min;
}
