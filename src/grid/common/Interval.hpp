#ifndef INTERVAL_HPP
#define INTERVAL_HPP

#include <algorithm>

class Interval {
		private:
				int xmin_{-1}, xmax_{-1};
				bool empty_{true};

		public:
				Interval() {
						empty_= true;
				}

				Interval(const int xmin, const int xmax) {
						xmin_ = xmin;
						xmax_ = xmax;
						empty_= false;
				}

				~Interval() {
				}

				inline void set_boundaries(const int xmin, const int xmax) {
						xmin_ = xmin;
						xmax_ = xmax;
						empty_= false;
				}

				inline bool is_point_in_interval(const int value) const {
						if (empty_)
								return false;
						return (value >= this->xmin_) && (value <= this->xmax_);
				}

				inline bool intersection_interval_is_empty(const Interval &y) const {
						/* return true if the intersection is empty */
						if (empty_ || y.empty_)
								return true;

						if ((this->xmin_ > y.xmax_) || (this->xmax_ < y.xmin_))
								return true;
						else
								return false;
				}

				inline Interval intersection_interval(const Interval &y) const {
						Interval z;
						if (!intersection_interval_is_empty(y)) {
								z.xmin_ = std::max(this->xmin_, y.xmin_);
								z.xmax_ = std::min(this->xmax_, y.xmax_);
						}
						return z;
				}

				inline int xmin() const {
						return xmin_;
				}

				inline int xmax() const {
						return xmax_;
				}
};
#endif
