#include "SparseMatrix.h"
#include "../Utils/Debug.h"
#include "NumericalMethodsTestAuto.h"
using namespace igmdk;

void testSimplex()
{
    Matrix<double> B = Matrix<double>::identity(3), N(3, 2);
    N(0, 0) = -2;
    N(0, 1) = 1;
    N(1, 0) = -1;
    N(1, 1) = 2;
    N(2, 0) = 1;
    N(2, 1) = 0;
    Vector<double> b, cB(3, 0), cN;
    b.append(2);
    b.append(7);
    b.append(3);
    cN.append(-1);
    cN.append(-2);
    LinearProgrammingSimplex s(B, N, cB, cN, b);
    Vector<pair<int, double> > result = s.solve();
    for(int i = 0; i < result.getSize(); ++i)
    {
        DEBUG(result[i].first);
        DEBUG(result[i].second);
    }
}

void testHessenbergPerm()
{//Permutation matrix
    Matrix<double> b2(3, 3);
    b2(0, 0) = 0;
    b2(0, 1) = 0;
    b2(0, 2) = 1;
    b2(1, 0) = 1;
    b2(1, 1) = 0;
    b2(1, 2) = 0;
    b2(2, 0) = 0;
    b2(2, 1) = 1;
    b2(2, 2) = 0;
    b2.debug();
    Vector<complex<double> > eig = QREigenHessenberg(b2);
    DEBUG("eig");
    eig.debug();
}

void testHessenbergComplex()
{//From Datta?
    Matrix<double> b2(3, 3);
    b2(0, 0) = 1;
    b2(0, 1) = 2;
    b2(0, 2) = 3;
    b2(1, 0) = 1;
    b2(1, 1) = 0;
    b2(1, 2) = 1;
    b2(2, 0) = 0;
    b2(2, 1) = -2;
    b2(2, 2) = 2;
    b2.debug();
    pair<Vector<complex<double> >, Matrix<double> > result = QREigen(b2);
    Vector<complex<double> > eig = result.first;
    DEBUG("eig");
    eig.debug();
    DEBUG("eigve");
    result.second.debug();
}

void testRepeatedNonsymmtric()
{
    int n = 3;
    Matrix<double> v(n, n), d = Matrix<double>::identity(n);
    d(0, 0) = 2;
    for(int i = 0; i < n; ++i)
    {
        Vector<double> vi = GlobalRNG().randomUnitVector(n);
        for(int j = 0; j < n; ++j) v(i, j) = vi[j];
    }
    LUP<double> lup(v);
    Matrix<double> b2 = v * d * inverse(lup, n);
    b2.debug();
    pair<Vector<complex<double> >, Matrix<double> > result = QREigen(b2);
    Vector<complex<double> > eig = result.first;
    DEBUG("eig");
    eig.debug();
    DEBUG("eigve");
    result.second.debug();
    Matrix<double> D(eig.getSize(), eig.getSize());
    for(int i = 0; i < eig.getSize(); ++i) D(i, i) = eig[i].real();
    QRDecomposition qr(result.second);
    DEBUG("V-1DV");
    (result.second * D * inverse(qr, n)).debug();
}

void testTridiagonalSparse2()
{
    //From Burden - correct
    SparseMatrix<double> b2(3, 3);
    b2.set(0, 0, 3);
    b2.set(0, 1, 1);
    b2.set(0, 2, 0);
    b2.set(1, 0, 1);
    b2.set(1, 1, 3);
    b2.set(1, 2, 1);
    b2.set(2, 0, 0);
    b2.set(2, 1, 1);
    b2.set(2, 2, 3);
    DEBUG("b2");
    b2.debug();
    pair<Vector<double>, Matrix<double> > EQ = LanczosEigenSymmetric(b2, 2);
    Vector<double> eig = EQ.first;
    DEBUG("eigVa");
    eig.debug();
    DEBUG("eigVe");
    EQ.second.debug();
    Matrix<double> D(eig.getSize(), eig.getSize());
    for(int i = 0; i < eig.getSize(); ++i) D(i, i) = eig[i];
    DEBUG("VTDV");
    (EQ.second.transpose() * D * EQ.second).debug();
}

void testTridiagonalSparse()
{
    //From Burden - correct
    SparseMatrix<double> b2(3, 3);
    b2.set(0, 0, 3);
    b2.set(0, 1, 1);
    b2.set(0, 2, 0);
    b2.set(1, 0, 1);
    b2.set(1, 1, 3);
    b2.set(1, 2, 1);
    b2.set(2, 0, 0);
    b2.set(2, 1, 1);
    b2.set(2, 2, 3);
    DEBUG("b2");
    b2.debug();
    BandMatrix<5> t2 = LanczosEigReduce(b2);
    DEBUG("t2");
    t2.debug();
    //Matrix<double> z2 = toDense(t2);
    //pair<Vector<double>, Matrix<double> > EQ = QREigenSymmetric(z2);
    Vector<double> eig = QREigenTridiagonal(t2);
    DEBUG("eigVa");
    eig.debug();
}

int main()
{
    matrixTestAllAuto();
    return 0;
    testHessenbergPerm();
    return 0;
    testHessenbergComplex();
    return 0;
    testTridiagonalSparse();
    return 0;
    testRepeatedNonsymmtric();
    return 0;
    testTridiagonalSparse2();
    return 0;


    testSimplex();
    return 0;
}
