#include "Contact.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    auto pd = Clean(CreatePolyData({{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}}));
    Contact contact(pd, pd);
    double origin[] = {0, 0, 0};
    double axis[] = {1, 0, 0};
    double normal[] = {0, 0, 1};
    Base2 base(origin, axis, normal);

    for (double eps : {1e-7, 3e-7, 9e-7}) {
        for (double y : {-.023, .023}) {
            Poly poly{{1, .00065, 0, 0}, {0, -eps, 0, 1}, {.2, y, 0, 2}};
            InterPtsType points;
            if (!contact.InterPolyLine(points, base, poly, Src::A) || points.size() != 2) {
                std::cerr << "Expected two finite-edge crossings for eps=" << eps << ", y=" << y << '\n';
                return EXIT_FAILURE;
            }

            double first = y < 0 ? eps/(.00065+eps) : .2*eps/(y+eps);
            double last = y < 0 ? .2+.8*(-y)/(.00065-y) : eps/(.00065+eps);
            if (std::abs(points.front().t-first) > 1e-12 ||
                std::abs(points.back().t-last) > 1e-12) {
                std::cerr << "Incorrect finite-edge crossing coordinates\n";
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}
