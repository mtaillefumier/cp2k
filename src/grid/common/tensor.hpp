#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <iostream>

template<typename T, int dim__> class tensor1 {
private:
		/// Size of the tensor1
		std::vector<int> size_;
		size_t full_size_ = {0};
		/// Leading dimension
		int ld_ = 0;
		T *data_host_{nullptr};
		T *data_gpu_{nullptr};
		/// set to true if memory is not allocated by the class
		bool allocated_externally_{false};
		/// size if the host buffer
		size_t allocated_size_{0};
public:
		tensor1(tensor1<T, dim__> const& src) = default;
		tensor1<T, dim__>&
		operator=(tensor1<T, dim__> const& src) = default;

		/// Move assigment operator
		inline tensor1<T, dim__>&
		operator=(tensor1<T, dim__> &&src)
				{
						if (this != &src) {
								size_ = src.size_;
								ld_ = src.ld_;
								data_host_ = src.data_host_;
								full_size_ = src.full_size_;
								allocated_externally_ = src.allocated_externally_;
								allocated_size_ = src.allocated_size_;
								src.data_host_ = nullptr;
						}
						return *this;
				}
		///Move constructor
		tensor1(tensor1<T, dim__> &&src)
				{
						size_ = src.size_;
						ld_ = src.ld_;
						data_host_ = src.data_host_;
						full_size_ = src.full_size_;
						allocated_externally_ = src.allocated_externally_;
						allocated_size_ = src.allocated_size_;
						src.data_host_ = nullptr;
				}


		~tensor1() {
				size_.clear();
				if (!allocated_externally_ && data_host_) {
						free(data_host_);
				}
		}

		/// release allocated memory but does not reset the size_ vector.
		void clear() {
				if (!allocated_externally_ && data_host_) {
						free(data_host_);
						allocated_size_ = 0;
						full_size_ = 0;
						// put the size of the tensor1 to 0
						for (auto &v : size_)
								v = 0;
				}
				data_host_ = nullptr;
		}

		tensor1() {
				size_.clear();
				size_.resize(dim__, 0);
		}

		tensor1(int x) {
				assert(dim__ == 1);
				assert(x > 0);
				size_.resize(dim__, 0);
				ld_ = x;
				need_to_reallocate();
		}

		tensor1(int x, int y) {
				assert(dim__ == 2);
				assert(x > 0);
				assert(y > 0);
				size_.clear();
				size_.resize(dim__, 0);
				size_[0] = x;
				size_[1] = y;
				ld_ = y;
				need_to_reallocate();
		}

		tensor1(int x, int y, int z) {
				assert(dim__ == 3);
				assert(x > 0);
				assert(y > 0);
				assert(z > 0);
				size_.clear();
				size_.resize(dim__, 0);
				size_[0] = x;
				size_[1] = y;
				size_[2] = z;
				ld_ = z;
				need_to_reallocate();
		}

		tensor1(int x, int y, int z, int w) {
				assert(dim__ == 4);
				assert(x > 0);
				assert(y > 0);
				assert(z > 0);
				assert(w > 0);
				size_.clear();
				size_.resize(dim__, 0);
				size_[0] = x;
				size_[1] = y;
				size_[2] = z;
				size_[3] = w;
				ld_ = w;
				need_to_reallocate();
		}

		tensor1(T *ptr, int x) {
				assert(dim__ == 1);
				size_.clear();
				size_.resize(dim__, 0);
				data_host_ = ptr;
				ld_ = x;
				allocated_externally_ = true;
		}

		tensor1(T *ptr, int x, int y) {
				assert(dim__ == 2);
				size_.clear();
				size_.resize(dim__, 0);
				size_[0] = x;
				size_[1] = y;
				ld_ = y;
				data_host_ = ptr;
				allocated_externally_ = true;
		}

		tensor1(T *ptr, int x, int y, int z) {
				assert(dim__ == 3);
				size_.clear();
				size_.resize(dim__, 0);
				size_[0] = x;
				size_[1] = y;
				size_[2] = z;
				ld_ = z;
				data_host_ = ptr;
				allocated_externally_ = true;
		}

		tensor1(T *ptr, int x, int y, int z, int w) {
				assert(dim__ == 4);
				size_.clear();
				size_.resize(dim__, 0);
				size_[0] = x;
				size_[1] = y;
				size_[2] = z;
				size_[3] = w;
				ld_ = w;
				data_host_ = ptr;
				allocated_externally_ = true;
		}

		void update_pointer(T *ptr) {
				if (!allocated_externally_) {
						if (data_host_)
								free(data_host_);
				}
				data_host_ = ptr;
				allocated_externally_ = true;
		}

		inline int size(const int i) const {
				assert(i < dim__);
				return size_[i];
		}
		/// return the leading dimenions
		inline size_t ld() const {
				return ld_;
		}

		/// resize a tensor1 of rank 1
		inline void resize(int x) {
				assert(dim__ == 1);
				assert(x >= 0);
				size_[0] = x;
				ld_ = x;
				need_to_reallocate();
		}
		/// resize a tensor1 of rank 2
		inline void resize(int x, int y) {
				assert(dim__ == 2);
				assert(x >= 0);
				assert(y >= 0);
				size_[0] = x;
				size_[1] = y;
				ld_ = y;
				need_to_reallocate();
		}

		/// resize a tensor1 of rank 3
		inline void resize(int x, int y, int z) {
				assert(dim__ == 3);
				assert(x >= 0);
				assert(y >= 0);
				assert(z >= 0);
				size_[0] = x;
				size_[1] = y;
				size_[2] = z;
				ld_ = z;
				need_to_reallocate();
		}

		/// resize a tensor1 of rank 4
		inline void resize(int x, int y, int z, int w) {
				assert(dim__ == 4);
				assert(x >= 0);
				assert(y >= 0);
				assert(z >= 0);
				assert(w >= 0);
				size_[0] = x;
				size_[1] = y;
				size_[2] = z;
				size_[3] = w;
				ld_ = w;
				need_to_reallocate();
		}

		inline size_t size() const {
				return full_size_;
		}

		inline T& operator()(int x) {
				assert(dim__ == 1);
				assert((x >= 0) && (x < size_[0]));
				return data_host_[x];
		}

		inline T& operator()(int x, int y) {
				assert(dim__ == 2);
				if ((size_[0] == 0) || (size_[1] == 0)) {
						std::cout << "Error\n" << std::endl;
						abort();
				}
				assert((x >= 0) && (x < size_[0]));
				if ((y < 0) || (y >= size_[1])) {
						std::cout << y << " " << size_[1] << std::endl;
						abort();
				}
				assert((y >= 0) && (y < size_[1]));
				return data_host_[x * ld_ + y];
		}


		inline T& operator()(int x, int y, int z) {
				assert(dim__ == 3);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				assert((z >= 0) && (z < size_[2]));
				return data_host_[(x * size_[1] + y) * ld_ + z];
		}


		inline T& operator()(int x, int y, int z, int w) {
				assert(dim__ == 4);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				assert((z >= 0) && (z < size_[2]));
				assert((w >= 0) && (w < size_[3]));
				return data_host_[((x * size_[1] + y) * size_[2] + z) * ld_ + w];
		}

		inline const T operator()(int x) const {
				assert(dim__ == 1);
				assert((x >= 0) && (x < size_[0]));
				return data_host_[x];
		}

		inline const T operator()(int x, int y) const {
				assert(dim__ == 2);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				return data_host_[x * ld_ + y];
		}


		inline const T operator()(int x, int y, int z) const {
				assert(dim__ == 3);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				assert((z >= 0) && (z < size_[2]));
				return data_host_[(x * size_[1] + y) * ld_ + z];
		}


		inline const T operator()(int x, int y, int z, int w) const {
				assert(dim__ == 4);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				assert((z >= 0) && (z < size_[2]));
				assert((w >= 0) && (w < size_[3]));
				return data_host_[((x * size_[1] + y) * size_[2] + z) * ld_ + w];
		}


		/// return the address of the buffer where data are stored
		inline T *at() {
				return data_host_;
		}

		/// return the address of the element (x)
		inline T *at(int x) {
				assert(dim__ == 1);
				assert((x >= 0) && (x < size_[1]));
				return data_host_ + x;
		}
		/// return the address of the element (x, y)
		inline T *at(int x, int y) {
				assert(dim__ == 2);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				return data_host_ + x * ld_ + y;
		}
		/// return the address of the element (x, y, z)
		inline T *at(int x, int y, int z) {
				assert(dim__ == 3);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				assert((z >= 0) && (z < size_[2]));
				return data_host_ + (x * size_[1] + y) * ld_ + z;
		}
		/// return the address of the element (x, y, z, w)
		inline T *at(int x, int y, int z, int w) {
				assert(dim__ == 4);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				assert((z >= 0) && (z < size_[2]));
				assert((w >= 0) && (w < size_[3]));

				return data_host_ + ((x * size_[1] + y) * size_[2] + z) * ld_ + w;
		}

		/// return the address of the element (x)
		inline const T *at(int x) const {
				assert(dim__ == 1);
				assert((x >= 0) && (x < size_[0]));
				return data_host_ + x;
		}
		/// return the address of the element (x, y)
		inline const T *at(int x, int y) const {
				assert(dim__ == 2);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				return data_host_ + x * ld_ + y;
		}
		/// return the address of the element (x, y, z)
		inline const T *at(int x, int y, int z) const {
				assert(dim__ == 3);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				assert((z >= 0) && (z < size_[2]));
				return data_host_ + (x * size_[1] + y) * ld_ + z;
		}
		/// return the address of the element (x, y, z, w)
		inline const T *at(int x, int y, int z, int w) const {
				assert(dim__ == 4);
				assert((x >= 0) && (x < size_[0]));
				assert((y >= 0) && (y < size_[1]));
				assert((z >= 0) && (z < size_[2]));
				assert((w >= 0) && (w < size_[3]));
				return data_host_ + ((x * size_[1] + y) * size_[2] + z) * ld_ + w;
		}

		inline void set_leading_dimension(const int ld) {
				ld_ = ld;
		}
/// initialize all elements to zero
		inline void zero() {
				if (data_host_ && full_size_) {
						memset(data_host_, 0, sizeof(T) * size());
				}
		}
private:
		void calculate_size_() {
				full_size_ = 1;
				for (auto i = 0u; i < size_.size(); i++) {
						full_size_ *= size_[i];
				}
		}

		void allocate_on_host() {
				allocated_size_ = ((full_size_ / 64) + 1) * 64;
				data_host_ = (T *)malloc(sizeof(T) * allocated_size_);
				allocated_externally_ = false;
		}

		void need_to_reallocate() {
				calculate_size_();

				if (full_size_ == 0)
						return;

				if (!allocated_externally_) {
						if (data_host_ == nullptr) {
								allocated_size_ = ((full_size_ / 64) + 1) * 64;
								data_host_ = static_cast<T*>(malloc(sizeof(T) * allocated_size_));
								return;
						}

						if (allocated_size_ < full_size_) {
								allocated_size_ = ((full_size_ / 64) + 1) * 64;
								data_host_ = static_cast<T*>(realloc((void *)data_host_, sizeof(T) * allocated_size_));
						}
				}
		}
};
#endif
