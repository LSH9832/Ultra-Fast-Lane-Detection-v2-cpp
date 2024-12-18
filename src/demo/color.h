#ifndef CV_DRAW_COLOR_H
#define CV_DRAW_COLOR_H

namespace cv {

    class _Color {
        const int _color_list[20][3] = {
            {255, 56, 56},
            {255, 157, 151},
            {255, 112, 31},
            {255, 178, 29},
            {207, 210, 49},
            {72, 249, 10},
            {146, 204, 23},
            {61, 219, 134},
            {26, 147, 52},
            {0, 212, 187},
            {44, 153, 168},
            {0, 194, 255},
            {52, 69, 147},
            {100, 115, 255},
            {0, 24, 236},
            {132, 56, 255},
            {82, 0, 133},
            {203, 56, 255},
            {255, 149, 200},
            {255, 55, 198}
        };

    public:
        template <class T>
        T as(int idx, bool bgr=true) {
            idx %= 20;
            if (bgr) return T(_color_list[idx][2], _color_list[idx][1], _color_list[idx][0]);
            else return T(_color_list[idx][0], _color_list[idx][1], _color_list[idx][2]);
        }

    };

    _Color color;
}


#endif