import numpy as np
cimport numpy as np
import math
import random
from cython import boundscheck, wraparound
from libc.math cimport cos, sin, signbit
from cython.parallel cimport prange, parallel


ctypedef double real_t
ctypedef double complex complex_t


cpdef real_t m_func(real_t x, real_t y, real_t R, real_t n_coef):
    cdef:
        real_t x_n = x * n_coef
        real_t y_n = y * n_coef
    if x_n * x_n + y_n * y_n <= R * R:
        return 1
    else:
        return 0

cpdef real_t p_func(real_t x, real_t y, real_t a):
    return a * (x * x + y * y)

cpdef real_t p_func3(real_t x, real_t y, real_t coef, real_t a_10, real_t a_01, real_t b_20, real_t b_11, real_t b_02, real_t c_30, real_t c_21, real_t c_12, real_t c_03):
    return coef * (a_10 * x + a_01 * y + b_20 * x * x + b_11 * x * y + b_02 * y * y + c_30 * x * x * x + c_21 * x * x * y + c_12 * x * y * y + c_03 * y * y * y)


@boundscheck(False)
@wraparound(False)
cpdef generate_psf(int dim, int m, real_t w_, real_t stationary_defocus):
    cdef:
        
        w = w_ * 2.34 * 0.001

        real_t f = 20 * 0.001
        real_t wavelength = 0.55 * 0.000001
        real_t d_1 = 57.4 * 0.001
        real_t d_0 = 37 * 0.001
        real_t r_0 = 4.5 * 0.001

        real_t lambda_d1 = wavelength * d_1
        real_t a = 2 * r_0 / lambda_d1
        real_t dzeta = 1.0 * w


        real_t ps_value = - np.pi * wavelength * d_1 * d_1 * dzeta / ((d_0 + w) * (d_0 + w))
        real_t n_coef = lambda_d1

        int dim_x = dim
        int dim_y = dim
        real_t min_x = -a
        real_t max_x = a
        real_t min_y = -a
        real_t max_y = a
        real_t range_x = max_x - min_x
        real_t range_y = max_y - min_y
        real_t R = r_0

        real_t delta_x = (max_x - min_x) / dim_x
        real_t delta_xi = 1.0 / dim_x / delta_x
        real_t dim_delta_xi = delta_xi * dim

        np.ndarray arg_x = np.linspace(start=min_x, stop=max_x, num=dim_x)
        np.ndarray arg_y = np.linspace(start=min_y, stop=max_y, num=dim_y)
        np.ndarray inner_h = np.zeros((dim_x, dim_y), dtype=np.cdouble)
        np.ndarray fft_inner_h = np.zeros((dim_x, dim_y), dtype=np.cdouble)
        real_t [:] x = arg_x
        real_t [:] y = arg_y
        complex_t [:,:] inner_h_view = inner_h
        complex_t [:,:] fft_inner_h_view = fft_inner_h

    cdef h = np.zeros((dim_x, dim_y, 2 * m - 1), dtype=np.double)

    for l in range(2 * m - 1):
        layer = l - m + 1
        for i in range(dim_x):
            for j in range(dim_y):
                temp = p_func(x[i], y[j], ps_value * (layer) / (m-1)) + p_func(x[i], y[j], ps_value * stationary_defocus)
                inner_h_view[i, j] = m_func(x[i], y[j], R, n_coef) * (cos(temp) + 1j * sin(temp))

        fft_inner_h = np.fft.fft2(inner_h)
        fft_inner_h = np.fft.fftshift(fft_inner_h)

        fft_inner_h = np.abs(fft_inner_h)
        fft_inner_h = fft_inner_h ** 2
        h[:, :, l] = np.real(fft_inner_h)

    h_max = np.amax(h)
    h = h / h_max

    return h


@boundscheck(False)
@wraparound(False)
cpdef generate_psf3(int dim, int m, real_t w_, real_t stationary_defocus, real_t a_10, real_t a_01, real_t b_20, real_t b_11, real_t b_02, real_t c_30, real_t c_21, real_t c_12, real_t c_03):
    cdef:

        w = w_ * 2.34 * 0.001

        real_t f = 20 * 0.001
        real_t wavelength = 0.55 * 0.000001
        real_t d_1 = 57.4 * 0.001
        real_t d_0 = 37 * 0.001
        real_t r_0 = 4.5 * 0.001


        real_t lambda_d1 = wavelength * d_1
        real_t a = 2 * r_0 / lambda_d1
        real_t dzeta = 1.0 * w


        real_t ps_value = - np.pi * wavelength * d_1 * d_1 * dzeta / ((d_0 + w) * (d_0 + w))
        real_t n_coef = lambda_d1

        int dim_x = dim
        int dim_y = dim
        real_t min_x = -a
        real_t max_x = a
        real_t min_y = -a
        real_t max_y = a
        real_t range_x = max_x - min_x
        real_t range_y = max_y - min_y
        real_t R = r_0

        real_t delta_x = (max_x - min_x) / dim_x
        real_t delta_xi = 1.0 / dim_x / delta_x
        real_t dim_delta_xi = delta_xi * dim

        np.ndarray arg_x = np.linspace(start=min_x, stop=max_x, num=dim_x)
        np.ndarray arg_y = np.linspace(start=min_y, stop=max_y, num=dim_y)
        np.ndarray inner_h = np.zeros((dim_x, dim_y), dtype=np.cdouble)
        np.ndarray fft_inner_h = np.zeros((dim_x, dim_y), dtype=np.cdouble)
        real_t [:] x = arg_x
        real_t [:] y = arg_y
        complex_t [:,:] inner_h_view = inner_h
        complex_t [:,:] fft_inner_h_view = fft_inner_h

    cdef h = np.zeros((dim_x, dim_y, 2 * m - 1), dtype=np.double)

    for l in range(2 * m - 1):
        layer = l - m + 1
        for i in range(dim_x):
            for j in range(dim_y):
                temp = p_func3(x[i], y[j], ps_value * (layer) / (m-1), a_10, a_01, b_20, b_11, b_02, c_30, c_21, c_12, c_03) + p_func3(x[i], y[j], ps_value * stationary_defocus, a_10, a_01, b_20, b_11, b_02, c_30, c_21, c_12, c_03)
                inner_h_view[i, j] = m_func(x[i], y[j], R, n_coef) * (cos(temp) + 1j * sin(temp))

        fft_inner_h = np.fft.fft2(inner_h)
        fft_inner_h = np.fft.fftshift(fft_inner_h)

        fft_inner_h = np.abs(fft_inner_h)
        fft_inner_h = fft_inner_h ** 2
        h[:, :, l] = np.real(fft_inner_h)

    h_max = np.amax(h)
    h = h / h_max

    return h


@boundscheck(False)
@wraparound(False)
cpdef generate_out_images(int dim, int m, real_t w_,  real_t stationary_defocus, np.ndarray[double, ndim=3] src):

    cdef int ext_dim = dim * 2 - 1
    cdef int cut_dim_s = dim / 2
    cdef int cut_dim_f = 3 * dim / 2
    cdef np.ndarray h = np.zeros((dim, dim, 2 * m - 1), dtype=np.double)
    cdef np.ndarray out = np.zeros((dim, dim, m), dtype=np.double)
    h = generate_psf(dim, m, w_, stationary_defocus)
    cdef np.ndarray ext_h = np.zeros((ext_dim, ext_dim, 2 * m - 1), dtype=np.complex)
    cdef np.ndarray ext_src = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)
    cdef np.ndarray ext_out = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)

    for l in range(m):
        ext_src[:dim, :dim, l] = src[:, :, l]
    for l in range(2 * m - 1):
        ext_h[:dim, :dim, l] = h[:, :, l]

    for l in range(m):
        ext_src[:, :, l] = np.fft.fft2(ext_src[:, :, l])
    for l in range(2 * m - 1):
        ext_h[:, :, l] = np.fft.fft2(ext_h[:, :, l])
        
    for k in range(m):
        for l in range(m):
            ext_out[:, :, k] = ext_out[:, :, k] + \
                               ext_src[:, :, l] * ext_h[:, :, l - k + m - 1]

    for l in range(m):
        ext_out[:, :, l] = np.fft.ifft2(ext_out[:, :, l])


    for l in range(m):
        out[:, :, l] = np.real(ext_out[cut_dim_s:cut_dim_f, cut_dim_s:cut_dim_f, l])

    out = out / np.amax(out)

    bytes_out = np.ndarray((dim, dim, m), np.ubyte)
    bytes_out = (255 * out).astype(np.ubyte)

    return out, bytes_out

@boundscheck(False)
@wraparound(False)
cpdef generate_out_images3(int dim, int m, real_t w_, real_t stationary_defocus, real_t a_10, real_t a_01, real_t b_20, real_t b_11, real_t b_02, real_t c_30, real_t c_21, real_t c_12, real_t c_03, np.ndarray[double, ndim=3] src):
    cdef int ext_dim = dim * 2 - 1
    cdef int cut_dim_s = dim / 2
    cdef int cut_dim_f = 3 * dim / 2
    cdef np.ndarray h = np.zeros((dim, dim, 2 * m - 1), dtype=np.double)
    cdef np.ndarray out = np.zeros((dim, dim, m), dtype=np.double)
    h = generate_psf3(dim, m, w_, stationary_defocus, a_10, a_01, b_20, b_11, b_02, c_30, c_21, c_12, c_03)
    cdef np.ndarray ext_h = np.zeros((ext_dim, ext_dim, 2 * m - 1), dtype=np.complex)
    cdef np.ndarray ext_src = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)
    cdef np.ndarray ext_out = np.zeros((ext_dim, ext_dim, m), dtype=np.complex)

    for l in range(m):
        ext_src[:dim, :dim, l] = src[:, :, l]
    for l in range(2 * m - 1):
        ext_h[:dim, :dim, l] = h[:, :, l]

    for l in range(m):
        ext_src[:, :, l] = np.fft.fft2(ext_src[:, :, l])
    for l in range(2 * m - 1):
        ext_h[:, :, l] = np.fft.fft2(ext_h[:, :, l])

    for k in range(m):
        for l in range(m):
            ext_out[:, :, k] = ext_out[:, :, k] + \
                               ext_src[:, :, l] * ext_h[:, :, l - k + m - 1]

    for l in range(m):
        ext_out[:, :, l] = np.fft.ifft2(ext_out[:, :, l])


    for l in range(m):
        out[:, :, l] = np.real(ext_out[cut_dim_s:cut_dim_f, cut_dim_s:cut_dim_f, l])

    out = out / np.amax(out)

    bytes_out = np.ndarray((dim, dim, m), np.ubyte)
    bytes_out = (255 * out).astype(np.ubyte)

    return out, bytes_out

@boundscheck(False)
@wraparound(False)
cpdef generate_out_images_gaussian_noise(int dim, int m, real_t w_, real_t stationary_defocus, np.ndarray[double, ndim=3] src, real_t noise_level):
    i = generate_out_images(dim, m, w_, stationary_defocus, src)[0]
    i_gaussian_noise = np.zeros((dim, dim, m))
    max_amp = np.max(i) - np.min(i)
    for j in range(m):
        for u in range(dim):
            for v in range(dim):
                i_gaussian_noise[u, v, j] = i[u, v, j] + random.gauss(0, 1) * max_amp * noise_level
    if np.min(i_gaussian_noise) < 0:
        i_gaussian_noise = i_gaussian_noise + abs(np.min(i_gaussian_noise))
    i_gaussian_noise = i_gaussian_noise / np.max(i_gaussian_noise)
    return i_gaussian_noise

@boundscheck(False)
@wraparound(False)
cpdef generate_out_images_shot_noise(int dim, int m, real_t w_, real_t stationary_defocus, np.ndarray[double, ndim=3] src, real_t k):
    i = generate_out_images(dim, m, w_, stationary_defocus, src)[0]
    i_shot_noise = np.zeros((dim, dim, m))
    for j in range(m):
        for u in range(dim):
            for v in range(dim):
                i_shot_noise[u, v, j] = np.random.poisson(abs(i[u, v, j] * k)) / k

    i_shot_noise = i_shot_noise / np.max(i_shot_noise)

    bytes_out = np.ndarray((dim, dim, m), np.ubyte)
    bytes_out = (255 * i_shot_noise).astype(np.ubyte)

    return i_shot_noise, bytes_out

@boundscheck(False)
@wraparound(False)
cpdef generate_out_images_shot_noise3(int dim, int m, real_t w_, real_t stationary_defocus, real_t a_10, real_t a_01, real_t b_20, real_t b_11, real_t b_02, real_t c_30, real_t c_21, real_t c_12, real_t c_03, np.ndarray[double, ndim=3] src, real_t k):
    i = generate_out_images3(dim, m, w_, stationary_defocus, a_10, a_01, b_20, b_11, b_02, c_30, c_21, c_12, c_03, src)[0]
    i_shot_noise = np.zeros((dim, dim, m))
    for j in range(m):
        for u in range(dim):
            for v in range(dim):
                i_shot_noise[u, v, j] = np.random.poisson(abs(i[u, v, j] * k)) / k

    i_shot_noise = i_shot_noise / np.max(i_shot_noise)

    bytes_out = np.ndarray((dim, dim, m), np.ubyte)
    bytes_out = (255 * i_shot_noise).astype(np.ubyte)

    return i_shot_noise, bytes_out

@boundscheck(False)
@wraparound(False)
cpdef generate_out_images_noise(int dim, int m, real_t w_, real_t stationary_defocus, np.ndarray[double, ndim=3] src, real_t noise_level, real_t k):
    i = generate_out_images(dim, m, w_, stationary_defocus, src)[0]   
    i_noise = np.zeros((dim, dim, m))
    max_amp = np.max(i) - np.min(i)
    for j in range(m):
        for u in range(dim):
            for v in range(dim):
                i_noise[u, v, j] = np.random.poisson(i[u, v, j] * k) / k + random.gauss(0, 1) * max_amp * noise_level
    i_noise = i_noise / np.max(i_noise)
    return i_noise