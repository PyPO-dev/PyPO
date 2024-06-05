#include <gtest/gtest.h>
#include "../Utils.h"
#include <array>
#include <cmath>
#define M_PI 3.14159265358979323846

class UtilsTest : public ::testing::Test {
    protected:
    
    std::complex<double> z0 = {1, 1};
    std::complex<double> z0c = {1, -1};
    std::complex<double> z0m = {-1, 1};
    std::complex<double> c0 = {0, 0};
    std::complex<double> j = {0, 1};
    
    Utils<double> ut_d;
    double mu = 1.;
    double err = 1e-16;

    std::array<double, 3> ad = {1, 1, 1};
    std::array<std::complex<double>, 3> cad = {z0, z0, z0};

    std::array<double, 3> xax = {1., 0, 0};
    std::array<double, 3> yax = {0, 1., 0};
    std::array<double, 3> zax = {0, 0, 1.};
    std::array<double, 3> zax_fl = {0, 0, -1.};

    std::array<double, 3> rot = {0., 0., 90 * M_PI / 180};
    std::array<double, 3> crot = {};

    double out_d;
    std::complex<double> cout_d;

    std::array<double, 3> ad_out;
    std::array<std::complex<double>, 3> cad_out;

    std::array<std::array<double, 3>, 3> md_out;
    std::array<std::array<double, 3>, 3> eye = {};
    double eye4[16] = {};

    void SetUp() override {
        this->eye[0][0] = 1.;
        this->eye[1][1] = 1.;
        this->eye[2][2] = 1.;
        
        this->eye4[0] = 1.;
        this->eye4[5] = 1.;
        this->eye4[10] = 1.;
        this->eye4[15] = 1.;
    }
};

TEST_F(UtilsTest, TestDotProducts) {
    ut_d.dot(ad, ad, out_d); // Check real-real dot
    EXPECT_EQ(out_d, 3.);
    
    ut_d.dot(cad, cad, cout_d);
    EXPECT_EQ(cout_d.real(), 6.);
    EXPECT_EQ(cout_d.imag(), 0.);

    ut_d.dot(cad, ad, cout_d);
    EXPECT_EQ(cout_d.real(), 3.);
    EXPECT_EQ(cout_d.imag(), -3.);
    
    ut_d.dot(ad, cad, cout_d);
    EXPECT_EQ(cout_d.real(), 3.);
    EXPECT_EQ(cout_d.imag(), 3.);
}

TEST_F(UtilsTest, TestExternalProducts) {
    ut_d.ext(ad, ad, ad_out); // Check parallel cross
    EXPECT_EQ(ad_out[0], 0.);
    EXPECT_EQ(ad_out[1], 0.);
    EXPECT_EQ(ad_out[2], 0.);

    ut_d.ext(xax, yax, ad_out); // Check x out y = z
    EXPECT_EQ(ad_out[0], zax[0]);
    EXPECT_EQ(ad_out[1], zax[1]);
    EXPECT_EQ(ad_out[2], zax[2]);
    
    ut_d.ext(yax, xax, ad_out); // Check y out x = -z
    EXPECT_EQ(ad_out[0], zax[0]);
    EXPECT_EQ(ad_out[1], zax[1]);
    EXPECT_EQ(ad_out[2], -zax[2]);
    
    ut_d.ext(cad, cad, cad_out); // Check parallel cross
    EXPECT_EQ(cad_out[0], c0);
    EXPECT_EQ(cad_out[1], c0);
    EXPECT_EQ(cad_out[2], c0);

    ut_d.ext(cad, xax, cad_out); // Check x out y = z
    EXPECT_EQ(cad_out[0], c0);
    EXPECT_EQ(cad_out[1], z0);
    EXPECT_EQ(cad_out[2], -z0);
    
    ut_d.ext(xax, cad, cad_out); // Check x out y = z
    EXPECT_EQ(cad_out[0], -c0);
    EXPECT_EQ(cad_out[1], -z0);
    EXPECT_EQ(cad_out[2], z0);
}

TEST_F(UtilsTest, TestVectorDifference) {
    ut_d.diff(ad, ad, ad_out);
    EXPECT_EQ(ad_out[0], 0.);
    EXPECT_EQ(ad_out[1], 0.);
    EXPECT_EQ(ad_out[2], 0.);

    ut_d.diff(xax, yax, ad_out);
    EXPECT_EQ(ad_out[0], 1.);
    EXPECT_EQ(ad_out[1], -1.);
    EXPECT_EQ(ad_out[2], 0);

    ut_d.diff(cad, cad, cad_out);
    EXPECT_EQ(cad_out[0], c0);
    EXPECT_EQ(cad_out[1], c0);
    EXPECT_EQ(cad_out[2], c0);
}

TEST_F(UtilsTest, TestAbsoluteValue) {
    ut_d.abs(ad, out_d);
    EXPECT_EQ(out_d, sqrt(3.));
    
    ut_d.abs(xax, out_d);
    EXPECT_EQ(out_d, 1.);
    
    ut_d.abs(yax, out_d);
    EXPECT_EQ(out_d, 1.);
    
    ut_d.abs(zax, out_d);
    EXPECT_EQ(out_d, 1.);
}

TEST_F(UtilsTest, TestNormalizeVector) {
    ut_d.normalize(ad, ad_out);
    ut_d.abs(ad_out, out_d);
    EXPECT_EQ(out_d, 1.);
}

TEST_F(UtilsTest, TestAddVectors) {
    ut_d.add(ad, ad, ad_out);
    EXPECT_EQ(ad_out[0], 2.);
    EXPECT_EQ(ad_out[1], 2.);
    EXPECT_EQ(ad_out[2], 2.);
}

TEST_F(UtilsTest, TestScalarMultiplication) {
    ut_d.s_mult(ad, 2., ad_out);
    EXPECT_EQ(ad_out[0], 2.);
    EXPECT_EQ(ad_out[1], 2.);
    EXPECT_EQ(ad_out[2], 2.);
    
    ut_d.s_mult(cad, j, cad_out);
    EXPECT_EQ(cad_out[0], z0m);
    EXPECT_EQ(cad_out[1], z0m);
    EXPECT_EQ(cad_out[2], z0m);
    
    ut_d.s_mult(ad, j, cad_out);
    EXPECT_EQ(cad_out[0], j);
    EXPECT_EQ(cad_out[1], j);
    EXPECT_EQ(cad_out[2], j);
    
    ut_d.s_mult(cad, 2., cad_out);
    EXPECT_EQ(cad_out[0], 2. * z0);
    EXPECT_EQ(cad_out[1], 2. * z0);
    EXPECT_EQ(cad_out[2], 2. * z0);
}

TEST_F(UtilsTest, TestConjugation) {
    ut_d.conj(cad, cad_out);
    EXPECT_EQ(cad_out[0], z0c);
    EXPECT_EQ(cad_out[1], z0c);
    EXPECT_EQ(cad_out[2], z0c);
}

TEST_F(UtilsTest, TestSnellReflection) {
    ut_d.snell(ad, zax_fl, ad_out);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], -ad[2]);
}

TEST_F(UtilsTest, TestSnellRefraction) {
    ut_d.snell_t(ad, zax, mu, ad_out);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], ad[2]);
}

TEST_F(UtilsTest, TestDyadicProducts) {
    ut_d.dyad(ad, ad, md_out);
    
    for(int n=0; n<3; n++) {
        EXPECT_EQ(md_out[n][0], 1.);
        EXPECT_EQ(md_out[n][1], 1.);
        EXPECT_EQ(md_out[n][2], 1.);
    }
}

TEST_F(UtilsTest, TestMatrixDifference) {
    ut_d.matDiff(eye, eye, md_out);
    
    for(int n=0; n<3; n++) {
        EXPECT_EQ(md_out[n][0], 0.);
        EXPECT_EQ(md_out[n][1], 0.);
        EXPECT_EQ(md_out[n][2], 0.);
    }
}

TEST_F(UtilsTest, TestMatrixMult) {
    ut_d.matVec(eye, ad, ad_out);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], ad[2]);
    
    ut_d.matVec(eye, cad, cad_out);
    EXPECT_EQ(cad_out[0], cad[0]);
    EXPECT_EQ(cad_out[1], cad[1]);
    EXPECT_EQ(cad_out[2], cad[2]);
}

TEST_F(UtilsTest, TestMatrixMult4D) {
    ut_d.matVec4(eye4, ad, ad_out);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], ad[2]);
}

TEST_F(UtilsTest, TestInverseMatrixMult4D) {
    ut_d.invmatVec4(eye4, ad, ad_out, true);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], ad[2]);
    
    ut_d.invmatVec4(eye4, ad, ad_out, false);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], ad[2]);
}

TEST_F(UtilsTest, TestMatrixRotation) {
    ut_d.matRot(rot, xax, crot, ad_out);
    EXPECT_NEAR(ad_out[0], yax[0], err);
    EXPECT_NEAR(ad_out[1], yax[1], err);
    EXPECT_NEAR(ad_out[2], yax[2], err);
}
