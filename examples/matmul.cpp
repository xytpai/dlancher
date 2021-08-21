#include <iostream>
#include "../dlancher.h"


int main()
{
    // create cpu memory
    float a[] = {
        -0.3902, -0.4178, -0.1564,  0.9405,  0.2758, -0.0184, -1.3352,  1.6319,
        -0.2514, -1.7879,  0.7837,  1.1945,  1.2567, -0.1773,  0.2808,  1.2030,
        -0.1633,  0.5050,  0.0874,  0.1481,  0.5652,  0.0810,  0.0494, -1.3074,
        -0.7059,  0.7443, -0.5154, -0.8449, -0.3103,  1.4287,  0.3826, -0.1402,
        -0.2242,  1.6692, -2.6425, -0.1756, -1.9548,  0.3375, -0.5413, -0.9390
    };
    float b[] = {
         1.1583,  0.3761,  0.7693,  0.0293,  0.6807, -2.0303, -0.0879,  1.0257,
         0.2069,  0.0968, -0.2670,  0.3872, -0.4218, -0.9323,  2.0011, -0.6779,
        -0.0574, -1.9720,  0.2780,  1.5022,  1.6819, -0.1036, -1.0977, -0.7408,
         0.1786,  0.9041,  0.9907,  0.6950, -0.1935, -0.0728,  1.0590,  0.8656
    };
    float m[] = {
        -0.4563,  0.2460, -0.6671, -0.3153, -1.4056, -0.1041,  0.0458,  0.1487,
         1.6991, -0.2064, -1.4353,  0.0631
    };
    int ah=5, aw=8;
    int bw=4;
    int mw=3;
    // copy to gpu memory
    int device = 0;
    int a_len = ah*aw;
    int b_len = aw*bw;
    int c_len = ah*bw;
    int cc_len = ah*mw;
    int m_len = bw*mw;
    dlancher::use(device);
    float *a_dev = dlancher::float32::malloc(device, a_len);
    float *b_dev = dlancher::float32::malloc(device, b_len);
    float *c_dev = dlancher::float32::malloc(device, c_len);
    float *cc_dev = dlancher::float32::malloc(device, cc_len);
    float *m_dev = dlancher::float32::malloc(device, m_len);
    dlancher::float32::memcpy(device, a_dev, -1, a, a_len);
    dlancher::float32::memcpy(device, b_dev, -1, b, b_len);
    dlancher::float32::memcpy(device, m_dev, -1, m, m_len);
    // kernel stream
    size_t stream;
    dlancher::create_stream(&stream);
    dlancher::float32::matmul(stream, a_dev, false, ah, aw, b_dev, false, bw, 
        c_dev, false, nullptr, nullptr);
    dlancher::float32::matmul(stream, c_dev, false, ah, bw, m_dev, false, mw, 
        cc_dev, false, nullptr, nullptr);
    // async and copy to cpu memory
    dlancher::async(stream);
    dlancher::float32::print2d(device, cc_dev, ah, mw);
    // reset device
    dlancher::reset();
    /*
    1.6006    3.1040    5.0529
    1.0389   -1.5508    8.8448
    0.3945    2.8369   -2.0581
   -1.1609    0.7573   -7.7238
    0.1465    5.2932   -5.0820
    */
    return 0;
}
