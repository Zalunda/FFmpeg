/*
 * RV60 dsp routines
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

#ifndef AVCODEC_RV60DSP_H
#define AVCODEC_RV60DSP_H

#include <stdint.h>

void ff_rv60_idct4x4_add(const int16_t * block, uint8_t * dst, int dst_stride);
void ff_rv60_idct8x8_add(const int16_t * block, uint8_t * dst, int dst_stride);
void ff_rv60_idct16x16_add(const int16_t * block, uint8_t * dst, int dst_stride);

#endif /* AVCODEC_RV60DSP_H */
