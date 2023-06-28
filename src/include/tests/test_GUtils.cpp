#include <gtest/gtest.h>
#include "../GUtils.h"
#include <array>
#include <cmath>
#include <cuComplex.h>
#define M_PI 3.14159265358979323846

class GUtilsTest : public ::testing::Test {
    protected:
    
    cuFloatComplex z0 = {1, 1};
    cuFloatComplex z0c = {1, -1};
    cuFloatComplex z0m = {-1, 1};
    cuFloatComplex c0 = {0, 0};
    cuFloatComplex j = {0, 1};
    
    float mu = 1.;
    float err = 1e-6;

    float ad [3]= {1, 1, 1};
    cuFloatComplex cad [3]= {z0, z0, z0};

    float xax[3] = {1., 0, 0};
    float yax[3] = {0, 1., 0};
    float zax[3] = {0, 0, 1.};
    float zax_fl[3] = {0, 0, -1.};

    float rot[3] = {0., 0., 90 * M_PI / 180};
    float crot[3] = {};

    float out_d;
    cuFloatComplex cout_d;

    float ad_out[3];
    cuFloatComplex cad_out[3];

    float md_out[3][3];
    float eye[3][3] = {};
    float eye4[16] = {};

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

bool equal(const cuFloatComplex &c1, const cuFloatComplex &c2) {
        if(c1.x == c2.x && c1.y == c2.y)
            return true;
        else
            return false;
}

TEST_F(GUtilsTest, TestDotProducts) {
    dot(ad, ad, out_d); // Check real-real dot
    EXPECT_EQ(out_d, 3.);
    
    dot(cad, cad, cout_d);
    EXPECT_EQ(cout_d.x, 6.);
    EXPECT_EQ(cout_d.y, 0.);

    dot(cad, ad, cout_d);
    EXPECT_EQ(cout_d.x, 3.);
    EXPECT_EQ(cout_d.y, -3.);
    
    dot(ad, cad, cout_d);
    EXPECT_EQ(cout_d.x, 3.);
    EXPECT_EQ(cout_d.y, 3.);
}

TEST_F(GUtilsTest, TestExternalProducts) {
    ext(ad, ad, ad_out); // Check parallel cross
    EXPECT_EQ(ad_out[0], 0.);
    EXPECT_EQ(ad_out[1], 0.);
    EXPECT_EQ(ad_out[2], 0.);

    ext(xax, yax, ad_out); // Check x out y = z
    EXPECT_EQ(ad_out[0], zax[0]);
    EXPECT_EQ(ad_out[1], zax[1]);
    EXPECT_EQ(ad_out[2], zax[2]);
    
    ext(yax, xax, ad_out); // Check y out x = -z
    EXPECT_EQ(ad_out[0], zax[0]);
    EXPECT_EQ(ad_out[1], zax[1]);
    EXPECT_EQ(ad_out[2], -zax[2]);
    
    ext(cad, cad, cad_out); // Check parallel cross
    EXPECT_TRUE(equal(cad_out[0], c0));
    EXPECT_TRUE(equal(cad_out[1], c0));
    EXPECT_TRUE(equal(cad_out[2], c0));

    ext(cad, xax, cad_out); // Check x out y = z
    EXPECT_TRUE(equal(cad_out[0], c0));
    EXPECT_TRUE(equal(cad_out[1], z0));
    EXPECT_TRUE(equal(cad_out[2], cuCmulf(make_cuFloatComplex(-1., 0), z0)));
    
    ext(xax, cad, cad_out); // Check x out y = z
    EXPECT_TRUE(equal(cad_out[0], cuCmulf(make_cuFloatComplex(-1., 0), c0)));
    EXPECT_TRUE(equal(cad_out[1], cuCmulf(make_cuFloatComplex(-1., 0), z0)));
    EXPECT_TRUE(equal(cad_out[2], z0));
}

TEST_F(GUtilsTest, TestVectorDifference) {
    diff(ad, ad, ad_out);
    EXPECT_EQ(ad_out[0], 0.);
    EXPECT_EQ(ad_out[1], 0.);
    EXPECT_EQ(ad_out[2], 0.);

    diff(xax, yax, ad_out);
    EXPECT_EQ(ad_out[0], 1.);
    EXPECT_EQ(ad_out[1], -1.);
    EXPECT_EQ(ad_out[2], 0);

    diff(cad, cad, cad_out);
    EXPECT_TRUE(equal(cad_out[0], c0));
    EXPECT_TRUE(equal(cad_out[1], c0));
    EXPECT_TRUE(equal(cad_out[2], c0));
}

TEST_F(GUtilsTest, TestAbsoluteValue) {
    abs(ad, out_d);
    EXPECT_EQ(out_d, (float)sqrt(3.));
    
    abs(xax, out_d);
    EXPECT_EQ(out_d, 1.);
    
    abs(yax, out_d);
    EXPECT_EQ(out_d, 1.);
    
    abs(zax, out_d);
    EXPECT_EQ(out_d, 1.);
}

TEST_F(GUtilsTest, TestNormalizeVector) {
    normalize(ad, ad_out);
    abs(ad_out, out_d);
    EXPECT_NEAR(out_d, 1., err);
}

TEST_F(GUtilsTest, TestAddVectors) {
    add(ad, ad, ad_out);
    EXPECT_EQ(ad_out[0], 2.);
    EXPECT_EQ(ad_out[1], 2.);
    EXPECT_EQ(ad_out[2], 2.);
}

TEST_F(GUtilsTest, TestScalarMultiplication) {
    float two = 2.;
    s_mult(ad, two, ad_out);
    EXPECT_EQ(ad_out[0], 2.);
    EXPECT_EQ(ad_out[1], 2.);
    EXPECT_EQ(ad_out[2], 2.);
    
    s_mult(cad, j, cad_out);
    EXPECT_TRUE(equal(cad_out[0], z0m));
    EXPECT_TRUE(equal(cad_out[1], z0m));
    EXPECT_TRUE(equal(cad_out[2], z0m));
    
    s_mult(ad, j, cad_out);
    EXPECT_TRUE(equal(cad_out[0], j));
    EXPECT_TRUE(equal(cad_out[1], j));
    EXPECT_TRUE(equal(cad_out[2], j));
    
    s_mult(cad, 2., cad_out);
    EXPECT_TRUE(equal(cad_out[0], cuCmulf(make_cuFloatComplex(2., 0), z0)));
    EXPECT_TRUE(equal(cad_out[1], cuCmulf(make_cuFloatComplex(2., 0), z0)));
    EXPECT_TRUE(equal(cad_out[2], cuCmulf(make_cuFloatComplex(2., 0), z0)));
}

TEST_F(GUtilsTest, TestConjugation) {
    conja(cad, cad_out);
    EXPECT_TRUE(equal(cad_out[0], z0c));
    EXPECT_TRUE(equal(cad_out[1], z0c));
    EXPECT_TRUE(equal(cad_out[2], z0c));
}

TEST_F(GUtilsTest, TestSnellReflection) {
    snell(ad, zax_fl, ad_out);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], -ad[2]);
}

TEST_F(GUtilsTest, TestSnellRefraction) {
    snell_t(ad, zax, mu, ad_out);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], ad[2]);
}

TEST_F(GUtilsTest, TestDyadicProducts) {
    dyad(ad, ad, md_out);
    
    for(int n=0; n<3; n++) {
        EXPECT_EQ(md_out[n][0], 1.);
        EXPECT_EQ(md_out[n][1], 1.);
        EXPECT_EQ(md_out[n][2], 1.);
    }
}

TEST_F(GUtilsTest, TestMatrixDifference) {
    matDiff(eye, eye, md_out);
    
    for(int n=0; n<3; n++) {
        EXPECT_EQ(md_out[n][0], 0.);
        EXPECT_EQ(md_out[n][1], 0.);
        EXPECT_EQ(md_out[n][2], 0.);
    }
}

TEST_F(GUtilsTest, TestMatrixMult) {
    matVec(eye, ad, ad_out);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], ad[2]);
    
    matVec(eye, cad, cad_out);
    EXPECT_TRUE(equal(cad_out[0], cad[0]));
    EXPECT_TRUE(equal(cad_out[1], cad[1]));
    EXPECT_TRUE(equal(cad_out[2], cad[2]));
}

TEST_F(GUtilsTest, TestMatrixMult4D) {
    matVec4(eye4, ad, ad_out);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], ad[2]);
}

TEST_F(GUtilsTest, TestInverseMatrixMult4D) {
    invmatVec4(eye4, ad, ad_out, true);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], ad[2]);
    
    invmatVec4(eye4, ad, ad_out, false);
    EXPECT_EQ(ad_out[0], ad[0]);
    EXPECT_EQ(ad_out[1], ad[1]);
    EXPECT_EQ(ad_out[2], ad[2]);
}

TEST_F(GUtilsTest, TestComplexExponential) {
    cuFloatComplex z_test = expCo(c0);
    EXPECT_EQ(z_test.x, 1.);
    EXPECT_EQ(z_test.y, 0.);
}
