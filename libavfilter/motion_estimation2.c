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

void ff_me_init_context2(AVMotionEstContext2 *me_ctx, int mb_size, int search_param,
                        int width, int height, int x_min, int x_max, int y_min, int y_max)
{
    me_ctx->width = width;
    me_ctx->height = height;
    me_ctx->mb_size = mb_size;
    me_ctx->search_param = search_param;
    me_ctx->get_cost = &ff_me_cmp_sad2;
    me_ctx->x_min = x_min;
    me_ctx->x_max = x_max;
    me_ctx->y_min = y_min;
    me_ctx->y_max = y_max;
}

uint32_t ff_me_cmp_sad2(AVMotionEstContext2 *me_ctx, int x_mb, int y_mb, int x_mv, int y_mv)
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

uint32_t ff_me_search_tdls2(AVMotionEstContext2 *me_ctx, int x_mb, int y_mb, int *mv)
{
    int x_min = FFMAX(me_ctx->x_min, x_mb - me_ctx->search_param);
    int y_min = FFMAX(me_ctx->y_min, y_mb - me_ctx->search_param);
    int x_max = FFMIN(x_mb + me_ctx->search_param, me_ctx->x_max);
    int y_max = FFMIN(y_mb + me_ctx->search_param, me_ctx->y_max);

    uint32_t cost;
    int step = ROUNDED_DIV(me_ctx->search_param, 2);
    uint32_t highest_costs[5] = {0, 0, 0, 0, 0};
    int cost_count = 0;

    mv[0] = x_mb;
    mv[1] = y_mb;
    mv[2] = UINT32_MAX;

    highest_costs[0] = me_ctx->get_cost(me_ctx, x_mb, y_mb, mv[0], mv[1]);

    do {
        int base_x = mv[0];
        int base_y = mv[1];

        for (int i = 0; i < 4; i++)
        {
            int x = base_x + dia1[i][0] * step;
            int y = base_y + dia1[i][1] * step;

            if (x >= x_min && x <= x_max && y >= y_min && y <= y_max)
            {
                cost = me_ctx->get_cost(me_ctx, x_mb, y_mb, x, y);
                if (cost < mv[2]) {
                    mv[0] = x;
                    mv[1] = y;
                    mv[2] = cost;
                }

                if (cost < highest_costs[4]) {
                    // Cost is too low to be in top 5, skip rest of comparisons
                } else if (cost < highest_costs[3]) {
                    highest_costs[4] = cost;
                } else if (cost < highest_costs[2]) {
                    highest_costs[4] = highest_costs[3];
                    highest_costs[3] = cost;
                } else if (cost < highest_costs[1]) {
                    highest_costs[4] = highest_costs[3];
                    highest_costs[3] = highest_costs[2];
                    highest_costs[2] = cost;
                } else if (cost < highest_costs[0]) {
                    highest_costs[4] = highest_costs[3];
                    highest_costs[3] = highest_costs[2];
                    highest_costs[2] = highest_costs[1];
                    highest_costs[1] = cost;
                } else {
                    highest_costs[4] = highest_costs[3];
                    highest_costs[3] = highest_costs[2];
                    highest_costs[2] = highest_costs[1];
                    highest_costs[1] = highest_costs[0];
                    highest_costs[0] = cost;
                }
            }
        }

        if (base_x == mv[0] && base_y == mv[1])
            step = step >> 1;

    } while (step > 0);

    mv[3] = (highest_costs[0] + highest_costs[1] + highest_costs[2] + highest_costs[3] + highest_costs[4]) / 5;
    if (mv[2] > 0 && (x_mb - mv[0] != 0 || y_mb - mv[1] != 0))
    {
        int k = 0;
    }

    return mv[2];
}
