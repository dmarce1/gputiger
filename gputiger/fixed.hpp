#pragma once

template<class T>
class fixed {
	T i;
	static constexpr float norm = 2.f * float(uint32_t(1) << uint32_t(31));
	static constexpr int mshift = 32;
public:
	fixed() = default;
	fixed(const fixed&) = default;
	fixed(fixed&&) = default;
	fixed& operator=(const fixed&) = default;
	fixed& operator=(fixed&&) = default;
	__device__ inline fixed(float r) {
		*this = r;
	}
	__device__   inline fixed& operator=(float r) {
		i = T(r * norm);
		return *this;
	}
	template<class V>
	__device__   inline fixed& operator=(const fixed<V>& r) {
		i = r.i;
		return *this;
	}

	template<class V>
	__device__   inline fixed(const fixed<V>& r) {
		*this = r;
	}

	__device__   inline fixed& operator+=(const fixed& other) {
		i += other.i;
		return *this;
	}

	__device__   inline fixed& operator-=(const fixed& other) {
		i -= other.i;
		return *this;
	}

	__device__   inline fixed& operator*=(const fixed& other) {
		i *= other.i;
		i >>= mshift;
		return *this;
	}

	__device__   inline fixed& operator/=(const fixed& other) {
		i /= (other.i >> mshift);
		return *this;
	}

	__device__   inline fixed operator+(const fixed& b) const {
		fixed c;
		c.i = i + b.i;
		return c;
	}
	__device__   inline fixed operator-(const fixed& b) const {
		fixed c;
		c.i = i - b.i;
		return c;
	}
	__device__   inline fixed operator*(const fixed& b) const {
		fixed c;
		c.i = (i * b.i) >> mshift;
		return c;
	}
	__device__   inline fixed operator/(const fixed& b) const {
		fixed c;
		c.i = i / (b.i >> mshift);
		return c;
	}

	__device__ inline
	float to_float() const {
		return float(i) / norm;
	}

	__device__ inline
	double to_double() const {
		return double(i) / norm;
	}

	template<class >
	friend class fixed;

	__device__ inline
	void set_integer(T j) {
		i = j;
	}

	template<class V>
	__device__ inline operator fixed<V>() const {
		fixed<V> v;
		v.i = i;
		return v;
	}
	__device__ inline
	bool operator>(fixed other) const {
		return i > other.i;
	}
	__device__ inline
	bool operator<(fixed other) const {
		return i <= other.i;
	}
	__device__ inline
	bool operator>=(fixed other) const {
		return i >= other.i;
	}
	__device__ inline
	bool operator<=(fixed other) const {
		return i <= other.i;
	}
	__device__ inline
	bool operator==(fixed other) const {
		return i == other.i;
	}
	__device__ inline
	bool operator!=(fixed other) const {
		return i != other.i;
	}

	__device__
	float ewald_dif(fixed other) {
		return ((int32_t) i - (int32_t) other.i) / norm;
	}

};

using fixed32 = fixed<uint32_t>;

using fixed64 = fixed<uint64_t>;
