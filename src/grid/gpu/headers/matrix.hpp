#ifndef MATRIX_HPP
#define MATRIX_HPP

class Matrix {
		const int col_{0};
		const int row_{0};
		const int ld_{0};
		double *ptr;

		Matrix(double *buffer, const int row__, const int col__) {
				ptr = buffer;
				row_ = row__;
				col_ = col__;
				ld_ = col__;
		}

		Matrix(double *buffer, const int row__, const int col__, const int ld__) {
				ptr = buffer;
				row_ = row__;
				col_ = col__;
				ld_ = ld__;
		}

		~Matrix() {
		}

		double operator() (int i, int j) {
				if ((i >= row_) || (j >= col_))
						return 0.0;
				return ptr[i * ld_ + j];
		}
		void set_element(int i, int j, double val) {
				if ((i >= row_) || (j >= col_))
						return;
				ptr[i * ld_ + j] = val;
		}
};
#endif
