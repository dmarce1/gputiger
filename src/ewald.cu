#include <gputiger/ewald.hpp>

__global__
void compute_ewald_table(ewald_table_t* table) {
	const int myindex = threadIdx.x + blockIdx.x * blockDim.x;
	const int blocksize = blockDim.x * gridDim.x;
	const float dx = 0.5f / float(EWALD_DIM - 1);
	const float fouroversqpi = 4.f / sqrtf(M_PI);
	const float twopi = 2.0f * M_PI;
	const float pi2over4 = M_PI * M_PI / 4.f;
	const float piover4 = M_PI / 4.f;
	const float piinv = 1.f / M_PI;
	float phi, fx, fy, fz;
	for (int index = myindex; index < EWALD_DIM3; index += blocksize) {
		int zi = index % EWALD_DIM;
		int yi = (index / EWALD_DIM) % EWALD_DIM;
		int xi = index / (EWALD_DIM * EWALD_DIM);
		float x = xi * dx;
		float y = yi * dx;
		float z = zi * dx;
		float r2 = x * x + y * y + z * z;
		if (r2 > 0.f) {
			fx = fy = fz = 0.f;
			phi = piover4;
			for (int nx = -4; nx <= +4; nx++) {
				for (int ny = -4; ny <= +4; ny++) {
					for (int nz = -4; nz <= +4; nz++) {
						float x0 = x - float(nx);
						float y0 = y - float(ny);
						float z0 = z - float(nz);
						float r2 = x0 * x0 + y0 * y0 + z0 * z0;
						if (r2 < 12.96f) {
							float r = sqrtf(r2);
							float rinv = rsqrtf(r2);
							float r2inv = rinv * rinv;
							float r3inv = r2inv * rinv;
							float erfc0 = erfcf(2.f * r);
							float exp0 = expf(-r * r);
							float d0 = -erfc0 * rinv;
							float d1 = (fouroversqpi * r * exp0 + erfc0) * r3inv;
							phi += d0;
							fx -= x0 * d1;
							fy -= y0 * d1;
							fz -= z0 * d1;
						}
					}
				}
			}
			for (int hx = -3; hx <= +3; hx++) {
				for (int hy = -3; hy <= +3; hy++) {
					for (int hz = -3; hz <= +3; hz++) {
						const int h2 = hx * hx + hy * hy + hz * hz;
						if (h2 < 10 && h2 != 0) {
							float hdotx = hx * x + hy * y + hz * z;
							float omega = twopi * hdotx;
							float s, c;
							sincosf(omega, &s, &c);
							float c0 = 1.f / float(h2) * expf(-pi2over4 * float(h2));
							float D0 = -piinv * c0;
							float Dx = 2.f * hx * c0;
							float Dy = 2.f * hy * c0;
							float Dz = 2.f * hz * c0;
							phi += D0 * c;
							fx -= Dx * s;
							fy -= Dy * s;
							fz -= Dz * s;
						}
					}
				}
			}
			float rinv = rsqrtf(r2);
			float r3inv = rinv * rinv * rinv;
			fx += x * r3inv;
			fy += y * r3inv;
			fz += z * r3inv;
			phi += rinv;
		} else {
			phi = 2.8372975f;
			fx = fy = fz = 0.f;
		}
		__syncthreads();
		(*table)[index][0] = fx;
		(*table)[index][1] = fy;
		(*table)[index][2] = fz;
		(*table)[index][3] = phi;
		__syncthreads();
	}
}
