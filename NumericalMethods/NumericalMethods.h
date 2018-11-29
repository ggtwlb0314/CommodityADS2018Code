#ifndef NUMERICAL_METHODS_H
#define NUMERICAL_METHODS_H
#include <cmath>
#include <complex>
#include <iomanip>
#include "../Utils/Bits.h"
#include "../Utils/Vector.h"
#include "../Utils/Utils.h"
#include "../Sorting/Sort.h"
#include "../Utils/Queue.h"
#include "../Heaps/Heap.h"
#include "../RandomNumberGeneration/Random.h"
#include "../RandomTreap/Treap.h"
#include "../ComputationalGeometry/Point.h"
#include "../Optimization/Metaheuristics.h"
#include "Matrix.h"
namespace igmdk{

bool haveDifferentSign(double a, double b){return (a < 0) != (b < 0);}
template<typename FUNCTION> pair<double, double> solveFor0(FUNCTION const& f,
    double xLeft, double xRight, double relAbsXPrecision = highPrecEps)
{
    double yLeft = f(xLeft), xMiddle = xLeft;
    assert(xRight >= xLeft && haveDifferentSign(yLeft, f(xRight)));
    while(isELess(xLeft, xRight, relAbsXPrecision))
    {
        xMiddle = (xLeft + xRight)/2;
        double yMiddle = f(xMiddle);
        if(haveDifferentSign(yLeft, yMiddle)) xRight = xMiddle;
        else
        {
            xLeft = xMiddle;
            yLeft = yMiddle;
        }
    }//best guess and worst-case error
    return make_pair((xLeft + xRight)/2, xRight - xLeft);
}
template<typename FUNCTION> pair<double, double> find1SidedInterval0(
    FUNCTION const& f, double x0 = 0, double d = 0.001, int maxEvals = 30)
{
    assert(maxEvals >= 2 && isfinite(x0 + d) && d != 0 &&
        !isEEqual(x0, x0 + d));
    for(double xLast = x0, f0 = f(x0); --maxEvals > 0; d *= 2)
    {
        double xNext = x0 + d, fNext = f(xNext);
        if(!isfinite(xNext) || isnan(fNext)) break;
        if(haveDifferentSign(f0, fNext))
            return d < 0 ? make_pair(xNext, xLast) : make_pair(xLast, xNext);
        xLast = xNext;
    }
    return make_pair(numeric_limits<double>::quiet_NaN(),
        numeric_limits<double>::quiet_NaN());
}
template<typename FUNCTION> pair<double, double> findInterval0(
    FUNCTION const& f, double x0, double d, int maxEvals)
{
    assert(maxEvals >= 2 && isfinite(x0 + d) && isELess(x0, x0 + d));
    for(double xLast = x0, f0 = f(x0); --maxEvals > 0; d = -d)
    {
        double xNext = x0 + d, fNext = f(xNext);
        if(!isfinite(xNext) || isnan(fNext)) break;
        if(haveDifferentSign(f0, fNext)) return d < 0 ?
            make_pair(xNext, x0 - (xLast - x0)) : make_pair(xLast, xNext);
        if(d < 0)
        {
            xLast = x0 - d;
            d *= 2;
        }
    }
    return make_pair(numeric_limits<double>::quiet_NaN(),
        numeric_limits<double>::quiet_NaN());
}
template<typename FUNCTION> pair<double, double> exponentialSearch(
    FUNCTION const& f, double x0 = 0, double step = 0.001,
    double xERelAbs = highPrecEps, int maxExpEvals = 60)
{
    pair<double, double> i0 = findInterval0(f, x0, step * max(1.0, abs(x0)),
        maxExpEvals);
    return !isnan(i0.first) ? solveFor0(f, i0.first, i0.second, xERelAbs) : i0;
}

Vector<complex<double> > findAllRoots(Vector<double> const& lowerCoefs)
{
    int n = lowerCoefs.getSize();
    Matrix<double> companion(n, n);
    for(int r = 0; r < n; ++r)
    {
        if(r > 0) companion(r, r - 1) = 1;
        companion(r, n - 1) = -lowerCoefs[r];
    }
    return QREigenHessenberg(companion);
}

template<typename FUNCTION> double estimateDerivativeFD(FUNCTION const& f,
    double x, double fx, double fEFactor = numeric_limits<double>::epsilon())
{
    double h = sqrt(fEFactor) * max(1.0, abs(x));
    return (f(x + h) - fx)/h;
}
template<typename FUNCTION> double estimateDerivativeCD(FUNCTION const& f,
    double x, double fEFactor = numeric_limits<double>::epsilon())
{
    double h = pow(fEFactor, 1.0/3) * max(1.0, abs(x));
    return (f(x + h) - f(x - h))/(2 * h);
}
//PRESENT?
template<typename FUNCTION> struct DerivFunctor
{
    FUNCTION f;
    DerivFunctor(FUNCTION const& theF): f(theF) {}
    double operator()(double p)const{return estimateDerivativeCD(f, p);}
    int fEvals()const{return 2;}
};

double findDirectionScale(Vector<double> const& x, Vector<double> u)
{
    for(int i = 0; i < x.getSize(); ++i) u[i] *= max(1.0, abs(x[i]));
    return norm(u);
}
template<typename FUNCTION> class ScaledDirectionFunction
{
    FUNCTION f;
    Vector<double> x, d;
    double scale;
public:
    ScaledDirectionFunction(FUNCTION const& theF, Vector<double> const& theX,
        Vector<double> const& theD): f(theF), x(theX), d(theD),
        scale(findDirectionScale(x, d * (1/norm(d)))){assert(norm(d) > 0);}
    double getS0()const{return scale;}//returns x[i] if x is axis vector
    double operator()(double s)const{return f(x + d * (s - getS0()));}
};
template<typename FUNCTION> Vector<double> estimateGradientCD(
    Vector<double> const& x, FUNCTION const& f,
    double fEFactor = numeric_limits<double>::epsilon())
{
    int D = x.getSize();
    Vector<double> result(D), d(D);
    for(int i = 0; i < D; ++i)
    {
        d[i] = 1;
        ScaledDirectionFunction<FUNCTION> df(f, x, d);
        result[i] = estimateDerivativeCD(df, df.getS0(), fEFactor);
        d[i] = 0;
    }
    return result;
}
template<typename FUNCTION> struct GradientFunctor
{
    FUNCTION f;
    GradientFunctor(FUNCTION const& theF): f(theF) {}
    Vector<double> operator()(Vector<double> const& p)const
        {return estimateGradientCD(p, f);}
    int fEvals(int D)const{return 2 * D;}
};

template<typename FUNCTION> double estimateDirectionalDerivativeCD(
    Vector<double> const& x, FUNCTION const& f, Vector<double> const& d,
    double fEFactor = numeric_limits<double>::epsilon())
{//estimates grad * d if d not unit
    ScaledDirectionFunction<FUNCTION> df(f, x, d);
    return estimateDerivativeCD(df, df.getS0(), fEFactor);
}

template<typename FUNCTION> struct DirectionalDerivativeFunctor
{
    FUNCTION f;
    DirectionalDerivativeFunctor(FUNCTION const& theF): f(theF) {}
    double operator()(Vector<double> const& x, Vector<double> const& d)
        const{return estimateDirectionalDerivativeCD(x, f, d);}
    int fEvals()const{return 2;}
};

template<typename FUNCTION>  Matrix<double> estimateJacobianCD(FUNCTION const&
    f, Vector<double> x, double fEFactor = numeric_limits<double>::epsilon())
{
    int n = x.getSize();
    Matrix<double> J(n, n);
    Vector<double> dx(n);
    double temp = pow(fEFactor, 1.0/3);
    for(int c = 0; c < n; ++c)
    {
        double xc = x[c], h = max(1.0, abs(xc)) * temp;
        x[c] += h;
        Vector<double> df = f(x);
        x[c] = xc - h;
        df -= f(x);
        x[c] = xc;
        for(int r = 0; r < n; ++r) J(r, c) = df[r]/(2 * h);
    }
    return J;
}

template<typename FUNCTION> double estimate2ndDerivativeCD(FUNCTION const& f,
    double x, double fx = numeric_limits<double>::quiet_NaN(),
    double fEFactor = numeric_limits<double>::epsilon())
{
    if(!isfinite(fx)) fx = f(x);
    double h = pow(fEFactor, 1.0/4) * max(1.0, abs(x));
    return (f(x + h) - 2 * fx + f(x - h))/(h * h);
}

//if gradient is estimated by finite diff, use
//fEFactor = pow(numeric_limits<double>::epsilon(), 2.0/3)
template<typename GRADIENT> Matrix<double> estimateHessianFromGradientCD(
    Vector<double> x, GRADIENT const& g,
    double fEFactor = numeric_limits<double>::epsilon())
{
    Matrix<double> HT = estimateJacobianCD(g, x, fEFactor);
    return 0.5 * (HT + HT.transpose());//ensure symmetry
}

template<typename DATA> class PiecewiseData
{
    typedef Treap<double, DATA> POINTS;
    mutable POINTS values;
    double eRelAbs;
public:
    int getSize()const{return values.getSize();}
    PiecewiseData(double theERelAbs = numeric_limits<double>::epsilon()):
        eRelAbs(theERelAbs){}
    typedef typename POINTS::NodeType NODE;
    pair<NODE*, NODE*> findPiece(double x)const
    {
        assert(!isnan(x));
        NODE* left = values.inclusivePredecessor(x), *right = 0;
        if(!left) right = values.findMin();
        else
        {
            typename POINTS::Iterator i(left);
            ++i;
            right = i != values.end() ? &*i : 0;
        }
        return make_pair(left, right);
    }
    NODE* eFind(double x)const
    {
        assert(!isnan(x));
        pair<NODE*, NODE*> piece = findPiece(x);
        if(piece.first && isEEqual(x, piece.first->key, eRelAbs))
            return piece.first;
        if(piece.second && isEEqual(x, piece.second->key, eRelAbs))
            return piece.second;
        return 0;
    }
    NODE* findMin()const
    {
        assert(values.getSize() > 0);
        return values.findMin();
    }
    NODE* findMax()const
    {
        assert(values.getSize() > 0);
        return values.findMax();
    }
    bool isInERange(double x)const
        {return getSize() > 1 && findMin()->key <= x && x <= findMax()->key;}
    void insert(double x, DATA const& y)
    {
        NODE* node = eFind(x);
        if(node) node->value = y;
        else values.insert(x, y);
    }
    void eRemove(double x)
    {
        NODE* node = eFind(x);
        if(node) values.removeFound(node);
    }
    Vector<pair<double, DATA> > getPieces()const
    {
        Vector<pair<double, DATA> > result;
        for(typename POINTS::Iterator iter = values.begin();
            iter != values.end(); ++iter)
            result.append(make_pair(iter->key, iter->value));
        return result;
    }
};

class DynamicLinearInterpolation
{
    PiecewiseData<double> pd;
public:
    DynamicLinearInterpolation(){}
    DynamicLinearInterpolation(Vector<pair<double, double> > const& xy)
    {//filter points to ensure eps x distance, use machine eps
        assert(isSorted(xy.getArray(), 0, xy.getSize() - 1,
            PairFirstComparator<double, double>()));
        for(int i = 0, lastGood = 0; i < xy.getSize(); ++i)
            if(i == 0 || isELess(xy[lastGood].first, xy[i].first))
            {
                insert(xy[i].first, xy[i].second);
                lastGood = i;
            }
    }
    double operator()(double x)const
    {
        if(!pd.isInERange(x)) return numeric_limits<double>::quiet_NaN();
        typedef typename PiecewiseData<double>::NODE NODE;
        pair<NODE*, NODE*> segment = pd.findPiece(x);
        assert(segment.first && segment.second);//sanity check
        double ly = segment.first->value, lx = segment.first->key;
        return ly + (segment.second->value - ly) * (x - lx)/
            (segment.second->key - lx);
    }
    void eRemove(double x){pd.eRemove(x);}
    bool eContains(double x){return pd.eFind(x) != 0;}
    void insert(double x, double y){pd.insert(x, y);}
};

class NotAKnotCubicSplineInterpolation
{//book M is a, book d is R
    struct Data{double a, b, c;};
    PiecewiseData<Data> pd;
public:
    NotAKnotCubicSplineInterpolation(Vector<pair<double, double> > xy,
        double eRelAbs = numeric_limits<double>::epsilon()): pd(eRelAbs)
    {//filter points to ensure eps x distance
        int n = xy.getSize(), skip = 0;
        assert(isSorted(xy.getArray(), 0, n - 1,
            PairFirstComparator<double, double>()));
        for(int i = 1; i + skip < n; ++i)
        {
            if(!isELess(xy[i - 1].first, xy[i + skip].first, eRelAbs)) ++skip;
            if(i + skip < n) xy[i] = xy[i + skip];
        }
        while(skip--) xy.removeLast();
        n = xy.getSize();
        assert(n > 3);//need 4 points to fit a cubic
        //special logic for endpoints
        double h1 = xy[1].first - xy[0].first, h2 = xy[2].first - xy[1].first,
            t1 = (xy[1].second - xy[0].second)/h1,
            t2 = (xy[2].second - xy[1].second)/h2,
            hnm2 = xy[n - 2].first - xy[n - 3].first,
            hnm1 = xy[n - 1].first - xy[n - 2].first,
            tnm2 = (xy[n - 2].second - xy[n - 3].second)/hnm2,
            tnm1 = (xy[n - 1].second - xy[n - 2].second)/hnm1;
        //take out points 1 and n - 2
        for(int i = 1; i < n - 3; ++i) xy[i] = xy[i + 1];
        xy[n - 3] = xy[n - 1];
        xy.removeLast();
        xy.removeLast();
        n = xy.getSize();
        //setup and solve tridiagonal system
        TridiagonalMatrix<double> T(
            2 * TridiagonalMatrix<double>::identity(n));
        Vector<double> R(n);
        //boundary conditions
        double D0Factor = 2/(2 * h2 + h1), Dnm1FActor = 2/(2 * hnm2 + hnm1);
        T(0, 1) = (2 * h1 + h2) * D0Factor;
        R[0] = 6 * (t2 - t1) * D0Factor;
        T(n - 1, n - 2) = (2 * hnm1 + hnm2) * Dnm1FActor;
        R[n - 1] = 6 * (tnm1 - tnm2) * Dnm1FActor;
        for(int i = 1; i < n - 1; ++i)
        {
            double hk = xy[i].first - xy[i - 1].first,
                tk = (xy[i].second - xy[i - 1].second)/hk,
                hkp1 = xy[i + 1].first - xy[i].first, hSum = hk + hkp1,
                tkp1 = (xy[i + 1].second - xy[i].second)/hkp1;
            R[i] = 6 * (tkp1 - tk)/hSum;
            T(i, i + 1) = hkp1/hSum;
            T(i, i - 1) = hk/hSum;
        }
        Vector<double> a = solveTridiag(T, R);
        //compute b and c
        for(int i = 1; i < n + 1; ++i)
        {
            double bi = 0, ci = 0;
            if(i < n)
            {
                double hi = xy[i].first - xy[i - 1].first;
                bi = (xy[i].second - xy[i - 1].second)/hi -
                    (a[i] - a[i - 1]) * hi/6;
                ci = xy[i - 1].second - a[i - 1] * hi * hi/6;
            }
            Data datai = {a[i - 1], bi, ci};
            pd.insert(xy[i - 1].first, datai);
        }
    }
    double operator()(double x, int deriv = 0)const
    {
        assert(deriv >= 0 && deriv <= 2);//support 2 continuous derivatives
        if(!pd.isInERange(x)) return numeric_limits<double>::quiet_NaN();
        typedef typename PiecewiseData<Data>::NODE NODE;
        pair<NODE*, NODE*> segment = pd.findPiece(x);
        assert(segment.first && segment.second);//sanity check
        double dxl = x - segment.first->key, dxr = segment.second->key - x,
            aim1 = segment.first->value.a, ai = segment.second->value.a,
            bi = segment.first->value.b, ci = segment.first->value.c,
            hi = dxr + dxl;
        if(deriv == 2) return (aim1 * dxr + ai * dxl)/hi;
        if(deriv == 1) return (-aim1 * dxr * dxr + ai * dxl * dxl)/(hi * 2) +
            bi;
        return (aim1 * dxr * dxr * dxr + ai * dxl * dxl * dxl)/(hi * 6) +
            bi * dxl + ci;
    }
};

class BarycentricInterpolation
{
    Vector<pair<double, double> > xy;
    Vector<double> w;
public:
    BarycentricInterpolation(Vector<pair<double, double> >const& thexy)
    {//O(n^2)
        for(int i = 0; i < thexy.getSize(); ++i)
            addPoint(thexy[i].first, thexy[i].second);
    }
    void addPoint(double x, double y)
    {
        double wProduct = 1;
        for(int i = 0; i < xy.getSize(); ++i)
        {
            wProduct *= (x - xy[i].first);
            w[i] /= xy[i].first - x;//must update previous wi
            assert(isfinite(w[i]));//signal repeated point or overflow
        }
        w.append(1/wProduct);
        assert(isfinite(w.lastItem()));
        xy.append(make_pair(x, y));
    }
    void removePoint(int i)
    {
        int n = xy.getSize();
        assert(i >= 0 && i < n);
        for(int j = 0; j < n; ++j)
            if(j != i) w[j] *= (xy[j].first - xy[i].first);
        for(int j = i + 1; j < n; ++j)
        {//make generic vector remove func as nonmember?
            xy[j - 1] = xy[j];
            xy.removeLast();
            w[j - 1] = w[j];
            w.removeLast();
        }
    }
    double operator()(double x)const
    {
        assert(isfinite(x));
        double numSum = 0, denomSum = 0;
        for(int i = 0; i < xy.getSize(); ++i)
        {
            double factorI = w[i]/(x - xy[i].first);
            if(!isfinite(factorI)) return xy[i].second;//inf if x is in xy
            numSum += factorI * xy[i].second;
            denomSum += factorI;
        }
        return numSum/denomSum;
    }
    double evalDeriv(double x)const
    {//unstable don't present!
        assert(isfinite(x));
        double numSum = 0, denomSum = 0, px = (*this)(x);
        for(int i = 0; i < xy.getSize(); ++i)
        {
            double factorI = w[i]/(x - xy[i].first);
            if(!isfinite(factorI))
            {//duplicates no problem
                numSum = 0;
                denomSum = w[i];
                for(int j = 0; j < xy.getSize(); ++j) if(j != i)
                {
                    double factorJ = w[j]/(xy[i].first - xy[j].first);
                    numSum += factorJ * (xy[j].second - xy[i].second);
                }
                break;
            }
            numSum += factorI * (px - xy[i].second)/(x - xy[i].first);
            denomSum += factorI;
        }
        return numSum/denomSum;
    }
    Matrix<double> diffMatrix()const
    {//duplicate points impossible here due to weight filtering
        int n = xy.getSize();
        assert(n > 1);//need at least two points for this to make sense
        Matrix<double> diff(n, n);
        for(int r = 0; r < n; ++r) for(int c = 0; c < n; ++c) if(r != c)
            {
                diff(r, c) = w[c]/w[r]/(xy[r].first - xy[c].first);
                diff(r, r) -= diff(r, c);
            }
        return diff;
    }
    Vector<double> getY()const
    {
        int n = xy.getSize();
        Vector<double> y(n);
        for(int i = 0; i < n; ++i) y[i] = xy[i].second;
        return y;
    }
    BarycentricInterpolation overwriteY(Vector<double> const& y)const
    {
        int n = y.getSize();
        assert(xy.getSize() == n);
        BarycentricInterpolation result = *this;
        for(int i = 0; i < n; ++i) {result.xy[i].second = y[i];}
        return result;
    }
    BarycentricInterpolation deriver()const
        {return overwriteY(diffMatrix() * getY());}
    BarycentricInterpolation integrator(int j = 0, double yj = 0)const
    {
        Vector<double> y = getY();
        Matrix<double> dm = diffMatrix();
        dm(j, j) += 1;
        y[j] += yj;
        double dmNorm = normInf(dm);
        //DEBUG(dmNorm);
        dm *= 1/dmNorm;
        y *= 1/dmNorm;
        QRDecomposition qr(dm);
        //LUP<> lup(dm);
        //DEBUG(lup.isSingular);
        //DEBUG("y");
        //y.debug();
        //DEBUG("dm");
        //dm.debug();
        //DEBUG("lup.d");
        //lup.d.debug();
        //DEBUG("solved y");
        //lup.solve(y).debug();
        //return overwriteY(lup.solve(y));
        //(qr.solve(y)).debug();
        return overwriteY(qr.solve(y));
    }
    double integrate()const
    {//assume sorted points
        BarycentricInterpolation i = integrator();
        return i.xy.lastItem().second;
    }
};

template<typename INTERPOLANT = BarycentricInterpolation>
class GenericPiecewiseInterpolation
{
    PiecewiseData<INTERPOLANT> pd;
    const INTERPOLANT* findInterpolant(double x)const
    {
        if(!pd.isInERange(x)) return 0;
        typename PiecewiseData<INTERPOLANT>::NODE* segment =
            pd.findPiece(x).first;
        assert(segment);//sanity check
        return &segment->value;
    }
public:
    GenericPiecewiseInterpolation(
        PiecewiseData<INTERPOLANT> const& thePd): pd(thePd){}
    double operator()(double x)const
    {
        const INTERPOLANT* i = findInterpolant(x);
        return i ? (*i)(x) : numeric_limits<double>::quiet_NaN();
    }
    double evalDeriv(double x)const
    {//unstable don't present!
        const INTERPOLANT* i = findInterpolant(x);
        return i ? (*i).evalDeriv(x) : numeric_limits<double>::quiet_NaN();
    }
    GenericPiecewiseInterpolation<INTERPOLANT> deriver()const
    {
        Vector<pair<double, INTERPOLANT> > pieces = pd.getPieces();
        PiecewiseData<INTERPOLANT> result;
        for(int i = 0; i < pieces.getSize(); ++i)//last one dummy
            result.insert(pieces[i].first, i < pieces.getSize() - 1 ?
                pieces[i].second.deriver() : pieces[i].second);
        return GenericPiecewiseInterpolation<INTERPOLANT>(result);
    }
    Vector<pair<pair<double, double>, INTERPOLANT> > getPieces()const
    {
        Vector<pair<double, INTERPOLANT> > pieces = pd.getPieces();
        assert(pieces.getSize() > 1);//1 real, 1 dummy
        Vector<pair<pair<double, double>, INTERPOLANT> > result;
        for(int i = 0; i < pieces.getSize(); ++i)
        {
            if(result.getSize() > 0)//set right boundary of prev piece
                result.lastItem().first.second = pieces[i].first;
            result.append(make_pair(make_pair(pieces[i].first, 0),
                pieces[i].second));
        }
        result.removeLast();//last piece is dummy
        return result;
    }
    double integrate()const
    {
        double sum = 0;
        Vector<pair<pair<double, double>, INTERPOLANT> > pieces = getPieces();
        for(int i = 0; i < pieces.getSize(); ++i)
            sum += pieces[i].second.integrate();
        return sum;
    }
};

complex<double> unityRootHelper(int j, int n)
    {return exp((j * PI()/n) * complex<double>(0, 1));}
Vector<complex<double> > FFTPower2(Vector<complex<double> > const&x)
{
    int n = x.getSize(), b = lgFloor(n);
    assert(isPowerOfTwo(n));
    typedef complex<double> C;
    Vector<C> result(n);
    for(unsigned int i = 0; i < n; ++i) result[reverseBits(i, b)] = x[i];
    for(int s = 1; s <= b; ++s)
    {
        int m = twoPower(s);
        C wm = unityRootHelper(-2, m);
        for(int k = 0; k < n; k += m)
        {
            C w(1, 0);
            for(int j = 0; j < m/2; ++j, w *= wm)
            {
                C t = w * result[k + j + m/2], u = result[k + j];
                result[k + j] = u + t;
                result[k + j + m/2] = u - t;
            }
        }
    }
    return result;
}

Vector<complex<double> > IFFTHelper(Vector<complex<double> > fftx)
{
    int n = fftx.getSize();
    fftx.reverse(1, n - 1);
    return fftx * (1.0/n);
}
Vector<complex<double> > inverseFFTPower2(
    Vector<complex<double> > const& x)
{
    assert(isPowerOfTwo(x.getSize()));
    return IFFTHelper(FFTPower2(x));
}

Vector<complex<double> > convolutionPower2(Vector<complex<double> > const& a,
    Vector<complex<double> > const& b)
{
    int n = a.getSize();
    assert(n == b.getSize() && isPowerOfTwo(n));
    Vector<complex<double> > fa = FFTPower2(a), fb = FFTPower2(b);
    for(int i = 0; i < n; ++i) fa[i] *= fb[i];
    return inverseFFTPower2(fa);
}
Vector<complex<double> > FFTGeneral(Vector<complex<double> > const& x)
{//Bluestein's algorithm
    int n = x.getSize(), m = nextPowerOfTwo(2 * n - 1);
    if(isPowerOfTwo(n)) return FFTPower2(x);
    Vector<complex<double> > a(m), b(m);//0-padded by default constructor
    for(int j = 0; j < n; ++j)
    {
        a[j] = x[j] * unityRootHelper(-j * j, n);
        b[j] = unityRootHelper(j * j, n);//could precompute b its fft
        if(j > 0) b[m - j] = b[j];
    }
    Vector<complex<double> > ab = convolutionPower2(a, b);
    while(ab.getSize() > n) ab.removeLast();
    for(int k = 0; k < n; ++k) ab[k] *= unityRootHelper(-k * k, n);
    return ab;
}
Vector<complex<double> > IFFTGeneral(Vector<complex<double> > const& x)
    {return IFFTHelper(FFTGeneral(x));}

pair<Vector<complex<double> >, Vector<complex<double> > > FFTReal2Seq(
    Vector<double> const& x, Vector<double> const& y)
{
    int n = x.getSize();
    assert(n == y.getSize());
    typedef complex<double> C;
    typedef Vector<C> VC;
    VC z(n);
    for(int i = 0; i < n; ++i) z[i] = C(x[i], y[i]);
    VC zf = FFTGeneral(z);
    pair<VC, VC> result;
    for(int i = 0; i < n; ++i)
    {
        C temp = conj(zf[(n - i) % n]);
        result.first.append(0.5 * (zf[i] + temp));
        result.second.append(0.5 * (zf[i] - temp) * C(0, -1));
    }
    return result;
}
Vector<complex<double> > FFTRealEven(Vector<double> const& x)
{
    int n = x.getSize(), n2 = n/2;
    assert(n % 2 == 0);
    typedef complex<double> C;
    typedef Vector<C> VC;
    Vector<double> xOdd(n2), xEven(n2);
    for(int i = 0; i < n; ++i) (i % 2 ? xOdd[i/2] : xEven[i/2]) = x[i];
    pair<VC, VC> xSplitF = FFTReal2Seq(xEven, xOdd);
    VC xF(n);
    C wn = unityRootHelper(-2, n), wi(1, 0);
    for(int i = 0; i < n2; ++i, wi *= wn)
    {
        xF[i] = xSplitF.first[i] + wi * xSplitF.second[i];
        xF[n2 + i] = xSplitF.first[i] - wi * xSplitF.second[i];
    }
    return xF;
}

Vector<double> DCTI(Vector<double> const& x)
{
    int n = x.getSize() - 1;
    assert(n > 0);
    Vector<double> y(2 * n), result(n + 1);
    for(int i = 0; i <= n; ++i) y[i] = x[i];
    for(int i = 1; i < n; ++i) y[2 * n - i] = x[i];
    Vector<complex<double> > yf = FFTRealEven(y);
    for(int i = 0; i <= n; ++i) result[i] = yf[i].real()/2;
    return result;
}
Vector<double> IDCTI(Vector<double> const& x)
    {return DCTI(x) * (2.0/(x.getSize() - 1));}

class ChebFunction
{
    Vector<double> ci;
    bool converged;
    double ciAbsE()const{return numeric_limits<double>::epsilon() *
        lgCeiling(ci.getSize()) * (1 + normInf(ci));}
    void trim()
    {//remove if ci too small
        int oldN = ci.getSize();
        double cutoff = ciAbsE();
        while(ci.getSize() > 1 && abs(ci.lastItem()) < cutoff)
            ci.removeLast();
        if(oldN - ci.getSize() > 1) converged = true;
    }
    //f values must be sorted values evaled at cos(jPi/n) for 0 <= j <= n
    void ChebFunctionHelper(Vector<double> const& fValues)
    {//DCTI does most work
        ci = DCTI(fValues) * (2.0/(fValues.getSize() - 1));
        ci[0] /= 2;//half first and last
        ci.lastItem() /= 2;
        converged = false;
        trim();
    }
public:
    ChebFunction(Vector<double> const& fValues, int n)
        {ChebFunctionHelper(fValues);}
    template<typename FUNCTION> ChebFunction(FUNCTION const& f, int n)
    {
        assert(n > 0);
        Vector<double> fValues(n + 1);
        for(int i = 0; i <= n; ++i)
            fValues[i] = f(cos(PI() * i/n));
        ChebFunctionHelper(fValues);
    }
    bool hasConverged()const{return converged;}
    double operator()(double x)const
    {
        assert(x >= -1 && x <= 1);
        double d = 0, dd = 0;
        for(int i = ci.getSize() - 1; i >= 0; --i)
        {
            double temp = d;
            d = (i == 0 ? 1 : 2) * x * d - dd + ci[i];
            dd = temp;
        }
        return d;
    }
    ChebFunction integral(double FM1 = 0)
    {//special case for 0 polynomial
        if(ci.getSize() == 1 && ci.lastItem() == 0) return *this;
        Vector<double> result;
        result.append(FM1);
        for(int i = 1; i - 1 < ci.getSize(); ++i)
            result.append(((i - 1 > 0 ? 1 : 2) * ci[i - 1] -
                (i + 1 > ci.getSize() - 1 ? 0 : ci[i + 1]))/2/i);
                ChebFunction cf(*this);
        cf.ci = result;
        cf.ci[0] -= cf(-1);
        return cf;
    }
    ChebFunction derivative()const
    {
        int n = ci.getSize() - 1;
        ChebFunction result(*this);
        if(n == 0) result.ci[0] = 0;
        else
        {
            result.ci = Vector<double>(n, 0);
            for(int i = n; i > 0; --i) result.ci[i - 1] =
                (i + 1 > n - 1 ? 0 : result.ci[i + 1]) + 2 * i * ci[i];
            result.ci[0] /= 2;
        }
        return result;
    }
    double error()const{return converged ? ciAbsE() : abs(ci.lastItem());}
    pair<double, double> integrate()const
    {
        double result = 0;
        for(int i = 0; i < ci.getSize(); i += 2)
            result += 2 * ci[i]/(1 - i * i);
        return make_pair(result, 2 * error());
    }
    Vector<double> findAllRealRoots(double complexAbsE = highPrecEps)const
    {
        int n = ci.getSize() - 1;
        if(n == 0) return Vector<double>(1, 0);//all 0 case
        else if(n == 1) return Vector<double>();//no roots constant poly
        //setup colleague matrix
        Matrix<double> colleague(n, n);//
        colleague(0, 1) = 1;
        for(int r = 1; r < n; ++r)
        {
            colleague(r, r - 1) = 0.5;
            if(r + 1 < n) colleague(r, r + 1) = 0.5;
            if(r == n - 1) for(int c = 0; c < n; ++c)
                colleague(r, c) -= ci[c]/(2 * ci[n]);
        }//solve + only keep real roots
        Vector<complex<double> > croots = QREigenHessenberg(colleague);
        Vector<double> result;//remove complex and extrapolated roots
        for(int i = 0; i < croots.getSize(); ++i) if(abs(croots[i].imag()) <
            complexAbsE && -1 <= croots[i].real() && croots[i].real() <= 1)
            result.append(croots[i].real());
        return result;
    }
    static double xToU(double x, double a, double b)
    {
        assert(a <= x && x <= b);
        double u = (2 * x - a - b)/(b - a);
        return u;
    }
    static double uToX(double u, double a, double b)
    {
        assert(-1 <= u && u <= 1);
        return ((b - a) * u + a + b)/2;
    }
};

template<typename FUNCTION>
Vector<double> reuseChebEvalPoints(FUNCTION const& f, Vector<double> const& fx)
{
    int n = 2 * (fx.getSize() - 1);
    assert(isPowerOfTwo(n));
    Vector<double> result;
    for(int i = 0; i <= n; ++i)
        result.append(i % 2 ? f(cos(PI() * i/n)) : fx[i/2]);
    return result;
}
template<typename FUNCTION> ChebFunction adaptiveChebEstimate(
    FUNCTION const& f, int maxEvals = 5000, int minEvals = 17)
{
    int n = minEvals - 1;
    assert(minEvals <= maxEvals && isPowerOfTwo(n));
    Vector<double> fx(n + 1);
    for(int i = 0; i <= n; ++i) fx[i] = f(cos(PI() * i/n));
    ChebFunction che(fx, n);
    while(maxEvals >= fx.getSize() + n && !che.hasConverged())
    {
        fx = reuseChebEvalPoints(f, fx);
        che = ChebFunction(fx, n *= 2);
    }
    return che;
}

template<typename FUNCTION> class ScaledFunctionM11
{//to allow [-1, 1] functions from any range
    FUNCTION f;
    double a, b;
public:
    ScaledFunctionM11(double theA, double theB, FUNCTION const& theF =
        FUNCTION()): f(theF), a(theA), b(theB) {assert(a < b);}
    double operator()(double u)const{return f(ChebFunction::uToX(u, a, b));}
};
struct ScaledChebAB
{//to eval Cheb at any range
    ChebFunction f;
    double a, b;
public:
    template<typename FUNCTION> ScaledChebAB(FUNCTION const& theF, int n,
        double theA, double theB): a(theA), b(theB),
        f(ScaledFunctionM11<FUNCTION>(theA, theB, theF), n) {assert(a < b);}
    ScaledChebAB(Vector<double> const& fValues, double theA, double theB):
        f(fValues, 0), a(theA), b(theB) {assert(a < b);}
    ScaledChebAB(ChebFunction const& theF, double theA, double theB):
        f(theF), a(theA), b(theB) {assert(a < b);}
    double operator()(double x)const{return f(ChebFunction::xToU(x, a, b));}
    pair<double, double> integrate()const
    {
        pair<double, double> result = f.integrate();
        result.first *= (b - a)/2;
        result.second *= (b - a)/2;
        return result;
    }
    double evalDeriv(double x)const
        {return 2/(b - a) * f.derivative()(ChebFunction::xToU(x, a, b));}
    Vector<double> findAllRealRoots()const
    {//default params good enough
        Vector<double> roots =  f.findAllRealRoots();
        for(int i = 0; i < roots.getSize(); ++i)
            roots[i] = ChebFunction::uToX(roots[i], a, b);
        return roots;
    }
};

class IntervalCheb
{
    ScaledChebAB cf;
    pair<double, double> ab;
    int maxEvals;
public:
    typedef ScaledChebAB INTERPOLANT;
    template<typename FUNCTION> IntervalCheb(FUNCTION const& f,
        double a, double b, int theMaxEvals = 64): ab(a, b),
        cf(f, theMaxEvals, a, b), maxEvals(theMaxEvals){}
    Vector<pair<double, ScaledChebAB> > getInterpolants()const
    {
        return Vector<pair<double, ScaledChebAB> >(1, make_pair(ab.first, cf));
    }
    double scaleEstimate()const{return 1;}
    double length()const{return 0;}
    double error()const//larger first
        {return cf.f.hasConverged() ? 0 : ab.second - ab.first;}
    template<typename FUNCTION>
    Vector<IntervalCheb> split(FUNCTION const& f)const
    {
        Vector<IntervalCheb> result;
        double middle = (ab.first + ab.second)/2;
        result.append(IntervalCheb(f, ab.first, middle, maxEvals));
        result.append(IntervalCheb(f, middle, ab.second, maxEvals));
        return result;
    }
    static int initEvals(int maxEvals){return maxEvals;}
    static int splitEvals(int maxEvals){return 2 * maxEvals;}
};

template<typename ADAPTIVE_INTERVAL> struct AdaptiveIntevalComparator
{
    double deltaLength;
    bool operator()(ADAPTIVE_INTERVAL const& lhs,
        ADAPTIVE_INTERVAL const& rhs)const
    {
        return (lhs.length() > deltaLength || rhs.length() > deltaLength) ?
            lhs.length() > rhs.length() : lhs.error() > rhs.error();
    }
};
template<typename INTERPOLATION_INTERVAL, typename FUNCTION>
    pair<GenericPiecewiseInterpolation<
    typename INTERPOLATION_INTERVAL::INTERPOLANT>,
    double> interpolateAdaptiveHeap(FUNCTION const& f, double a, double b,
    double param, double eRelAbs = highPrecEps, int maxEvals = 1000000,
    int minEvals = -1)
{
    typedef INTERPOLATION_INTERVAL II;
    typedef typename II::INTERPOLANT I;
    if(minEvals == -1) minEvals = sqrt(maxEvals);
    assert(a < b && maxEvals >= minEvals && minEvals >= II::initEvals(param));
    INTERPOLATION_INTERVAL i0(f, a, b, param);
    double scale = i0.scaleEstimate();
    AdaptiveIntevalComparator<II> ic = {(b - a)/minEvals};
    Heap<II, AdaptiveIntevalComparator<II> > h(ic);
    h.insert(i0);
    for(int usedEvals = II::initEvals(param);
        usedEvals < minEvals || (
        usedEvals + II::splitEvals(param) <= maxEvals &&
        !isEEqual(scale, scale + h.getMin().error(), eRelAbs));
        usedEvals += II::splitEvals(param))
    {
        II next = h.deleteMin();
        Vector<II> division = next.split(f);
        for(int i = 0; i < division.getSize(); ++i) h.insert(division[i]);
    }//process heap intervals
    PiecewiseData<I> pd;
    double error = h.getMin().error(),
        right = -numeric_limits<double>::infinity();
    while(!h.isEmpty())
    {
        II next = h.deleteMin();
        Vector<pair<double, I> > interpolants = next.getInterpolants();
        for(int i = 0; i < interpolants.getSize(); ++i)
            pd.insert(interpolants[i].first, interpolants[i].second);
    }//need dummy right endpoint interpolator
    pd.insert(b, i0.getInterpolants().lastItem().second);
    return make_pair(GenericPiecewiseInterpolation<I>(pd), error);
}

template<typename FUNCTION> pair<double, double> integrateCC(FUNCTION const& f,
    double a, double b, int maxEvals = 5000, int minEvals = 17)
{
    ScaledChebAB c(adaptiveChebEstimate(ScaledFunctionM11<FUNCTION>(a, b, f),
        maxEvals, minEvals), a, b);
    return c.integrate();
}

class IntervalGaussLobattoKronrod
{
    double a, b, y[7];
    void generateX(double x[7])const
    {
        x[0] = -1; x[1] = -sqrt(2.0/3); x[2] = -1/sqrt(5); x[3] = 0;
        x[4] = -x[2]; x[5] = -x[1]; x[6] = -x[0];
    }
    template<typename FUNCTION> initHelper(FUNCTION const& f, double fa,
        double fb)
    {
        assert(a < b);
        y[0] = fa;
        y[6] = fb;
        double x[7];
        generateX(x);
        ScaledFunctionM11<FUNCTION> fM11(a, b, f);
        for(int i = 1; i < 6; ++i) y[i] = fM11(x[i]);
    }
    double integrateGL()const
    {
        double w[7] = {1.0/6, 0, 5.0/6, 0};
        w[4] = w[2]; w[5] = w[1]; w[6] = w[0];
        double result = 0;
        for(int i = 0; i < 7; ++i) result += w[i] * y[i];
        return result * (b - a)/2;
    }
    template<typename FUNCTION> IntervalGaussLobattoKronrod(FUNCTION const& f,
        double theA, double theB, double fa, double fb): a(theA), b(theB)
        {initHelper(f, fa, fb);}
public:
    template<typename FUNCTION> IntervalGaussLobattoKronrod(FUNCTION const& f,
        double theA, double theB): a(theA), b(theB)
        {initHelper(f, f(a), f(b));}
    double integrate()const
    {
        double w[7] = {11.0/210, 72.0/245, 125.0/294, 16.0/35};
        w[4] = w[2]; w[5] = w[1]; w[6] = w[0];
        double result = 0;
        for(int i = 0; i < 7; ++i) result += w[i] * y[i];
        return result * (b - a)/2;//need to scale back
    }
    double length()const{return b - a;}
    double error()const{return abs(integrate() - integrateGL());}
    template<typename FUNCTION>
    Vector<IntervalGaussLobattoKronrod> split(FUNCTION const& f)const
    {
        Vector<IntervalGaussLobattoKronrod> result;
        double x[7];
        generateX(x);
        for(int i = 0; i < 7; ++i) x[i] = a + (x[i] - -1) * (b - a)/2;
        for(int i = 0; i < 6; ++i) result.append(
            IntervalGaussLobattoKronrod(f, x[i], x[i + 1], y[i], y[i + 1]));
        return result;
    }
    static int initEvals(){return 7;}
    static int splitEvals(){return 30;}
};

/*
Want the most robust integrator that does best precision in allowed num of
evals upto traget precision
So use abs precision not sum that cancels out in other rules and dont strive
for super small num of evals -- want precision instead to each small
subinterval must have small enough length
*/

template<typename INTEGRATION_INTERVAL, typename FUNCTION> pair<double, double>
    integrateAdaptiveHeap(FUNCTION const& f, double a, double b, double
    eRelAbs = highPrecEps, int maxEvals = 1000000, int minEvals = -1)
{
    typedef INTEGRATION_INTERVAL II;
    if(minEvals == -1) minEvals = sqrt(maxEvals);
    assert(a < b && maxEvals >= minEvals && minEvals >= II::initEvals());
    II i0(f, a, b);
    double result = i0.integrate(), totalError = i0.error();
    AdaptiveIntevalComparator<II> ic = {(b - a)/minEvals};
    Heap<II, AdaptiveIntevalComparator<II> > h(ic);
    h.insert(i0);
    for(int usedEvals = II::initEvals();
        usedEvals < minEvals || (usedEvals + II::splitEvals() <= maxEvals &&
        !isEEqual(result, result + totalError, eRelAbs));
        usedEvals += II::splitEvals())
    {
        II next = h.deleteMin();
        Vector<II> division = next.split(f);
        result -= next.integrate();
        totalError -= next.error();
        for(int i = 0; i < division.getSize(); ++i)
        {
            h.insert(division[i]);
            result += division[i].integrate();
            totalError += division[i].error();
        }
    }
    return make_pair(result, totalError);
}
template<typename FUNCTION> struct SingularityWrapper
{
    FUNCTION f;
    mutable int sCount;
    SingularityWrapper(FUNCTION const& theF = FUNCTION()): f(theF), sCount(0){}
    double operator()(double x)const
    {
        double y = f(x);
        if(isfinite(y)) return y;
        else
        {
            ++sCount;
            return 0;
        }
    }
};

template<typename FUNCTION> pair<double, double> integrateHybrid(
    FUNCTION const& f, double a, double b, double eRelAbs = highPrecEps,
    int maxEvals = 1000000, int minEvals = -1)
{
    int CCEvals = min(maxEvals/2, 1000);
    pair<double, double> resultCC = integrateCC(f, a, b, CCEvals);
    if(isEEqual(resultCC.first, resultCC.first + resultCC.second, eRelAbs))
        return resultCC;
    pair<double, double> resultGLK = integrateAdaptiveHeap<
        IntervalGaussLobattoKronrod>(f, a, b, eRelAbs, maxEvals - CCEvals,
        minEvals);
    return resultCC.second < resultGLK.second ? resultCC : resultGLK;
}

pair<double, double> integrateFromData(Vector<pair<double, double> > xyPairs)
{
    assert(xyPairs.getSize() >= 3);
    quickSort(xyPairs.getArray(), 0, xyPairs.getSize() - 1,
        PairFirstComparator<double, double>());
    double result[2] = {0, 0}, last = 0;
    for(int j = 2; j >= 1; --j)
        for(int i = 0; i + j < xyPairs.getSize(); i += j) result[j - 1] +=
            last = (xyPairs[i + j].first - xyPairs[i].first) *
            (xyPairs[i + j].second + xyPairs[i].second)/2;
    if(xyPairs.getSize() % 2 == 0) result[1] += last;
    return make_pair(result[0], abs(result[0] -  result[1]));
}

template<typename TWO_VAR_FUNCTION>
double RungKutta4Step(TWO_VAR_FUNCTION const& f, double x, double y,
    double h, double f0 = numeric_limits<double>::quiet_NaN())
{
    if(isnan(f0)) f0 = f(x, y);
    double k1 = h * f0, k2 = h * f(x + h/2, y + k1/2),
        k3 = h * f(x + h/2, y + k2/2), k4 = h * f(x + h, y + k3);
    return y + (k1 + 2 * k2 + 2 * k3 + k4)/6;
}

template<typename TWO_VAR_FUNCTION> pair<pair<double, double>, double>
    RungKuttaDormandPrinceStep(TWO_VAR_FUNCTION const& f, double x, double y,
    double h, double f0)
{
    double k1 = h * f0,
        k2 = h * f(x + h/5, y + k1/5),
        k3 = h * f(x + h * 3/10, y + k1 * 3/40 + k2 * 9/40),
        k4 = h * f(x + h * 4/5, y + k1 * 44/45 + k2 * -56/15 + k3 * 32/9),
        k5 = h * f(x + h * 8/9, y + k1 * 19372/6561 + k2 * -25360/2187 +
            k3 * 64448/6561 + k4 * -212/729),
        k6 = h * f(x + h, y + k1 * 9017/3168 + k2 * -355/33 + k3 * 46732/5247 +
            k4 * 49/176 + k5 * -5103/18656),
        yNew = y + k1 * 35/384 + k3 * 500/1113 + k4 * 125/192 +
            k5 * -2187/6784 + k6 * 11/84, f1 = f(x + h, yNew),
        k7 = h * f1;
    return make_pair(make_pair(yNew, f1), y + k1 * 5179/57600 + k3 * 7571/16695
        + k4 * 393/640 + k5 * -92097/339200 + k6 * 187/2100 + k7/40);
}
template<typename TWO_VAR_FUNCTION> pair<double, double>
    adaptiveRungKuttaDormandPrice(TWO_VAR_FUNCTION const& f, double x0,
    double xGoal, double y0, double localERelAbs = defaultPrecEps,
    int maxIntervals = 100000, int minIntervals = -1, int upSkip = 5)
{
    if(minIntervals == -1) minIntervals = sqrt(maxIntervals);
    assert(xGoal > x0 && minIntervals > 0 && upSkip > 0);
    double hMax = (xGoal - x0)/minIntervals, hMin = (xGoal - x0)/maxIntervals,
        linearError = 0, h1 = hMax, y = y0, f0 = f(x0, y);
    bool last = false;
    int stepCounter = 0;
    for(double x = x0; !last;)
    {
        if(x + h1 > xGoal)
        {//make last step accurate
            h1 = xGoal - x;
            last = true;
        }
        pair<pair<double, double>, double> yfye =
            RungKuttaDormandPrinceStep(f, x, y, h1, f0);
        double h2 = h1/2, xFraction = h1/(xGoal - x0);
        if(h2 < hMin || isEEqual(yfye.first.first, yfye.second,
            max(highPrecEps, localERelAbs * sqrt(xFraction))))
        {//accept step
            x += h1;
            y = yfye.first.first;
            f0 = yfye.first.second;//reuse last eval
            linearError += abs(y - yfye.second);
            if(++stepCounter == upSkip && h2 >= hMin)
            {//use larger step after few consecutive accepted steps
                h1 = min(hMax, h1 * 2);
                stepCounter = 0;
            }
        }
        else
        {//use half step
            h1 = h2;
            last = false;
            stepCounter = 0;
        }
    }
    return make_pair(y, linearError);
}

struct MultivarFuncHelper
{
    struct F1DBase
        {virtual double operator()(Vector<double> const& x)const = 0;};
    Vector<F1DBase*> fs;//beware storage is elsewhere
    Vector<double> operator()(Vector<double> const& x)const
    {
        Vector<double> y(fs.getSize());
        for(int i = 0; i < fs.getSize(); ++i) y[i] = (*fs[i])(x);
        return y;
    }
};

double normInf(double x){return abs(x);}
template<typename FUNCTION, typename X> bool equationBacktrack(
    FUNCTION const& f, X& x, X& fx, int& maxEvals, X const& dx, double xEps)
{
    bool failed = true;
    for(double s = 1;
        maxEvals > 0 && normInf(dx) * s > xEps * (1 + normInf(x)); s /= 2)
    {
        X fNew = f(x + dx * s);
        --maxEvals;
        if(normInf(fNew) <= (1 - 0.0001 * s) * normInf(fx))
        {
            failed = false;
            x += dx * s;
            fx = fNew;
            break;
        }
    }
    return failed;
}

template<typename X> struct BroydenSecant
{
    static int getD(double dummy){return 1;}
    static double generateUnitStep(double dummy, double infNormSize)
        {return GlobalRNG().normal(0, infNormSize);}
    class InverseOperator
    {
        double b;
    public:
        template<typename FUNCTION> InverseOperator(FUNCTION const& f,
            double x): b(estimateDerivativeCD(f, x)){}
        void addUpdate(double df, double dx){b = df/dx;}
        double operator*(double fx)const{return fx/b;}
    };
};
template<> struct BroydenSecant<Vector<double> >
{
    typedef Vector<double> X;
    static int getD(X const& x){return x.getSize();}
    static X generateUnitStep(X const& x, double infNormSize)
    {
        return GlobalRNG().randomUnitVector(getD(x)) *
            GlobalRNG().normal(0, infNormSize);
    }
    class InverseOperator
    {
        QRDecomposition qr;
    public:
        template<typename FUNCTION> InverseOperator(FUNCTION const& f,
            X const& x): qr(estimateJacobianCD(f, x)){}
        void addUpdate(X const& df, X dx)
        {
            double ndx2 = norm(dx);
            dx *= (1/ndx2);
            qr.rank1Update(df * (1/ndx2) - qr * dx, dx);
        }
        X operator*(X const& fx)const{return qr.solve(fx);}
    };
};
template<typename FUNCTION, typename X> bool equationTryRandomStep(
    FUNCTION const& f, X& x, X& fx, double stepNorm)
{
    X dx = BroydenSecant<X>::generateUnitStep(x, stepNorm),
        fNew = f(x + dx);
    bool improved = normInf(fNew) < normInf(fx);
    if(improved)
    {
        x += dx;
        fx = fNew;
    }
    return !improved;
}

template<typename FUNCTION, typename X> pair<X, double> solveBroyden(
    FUNCTION const& f, X const& x0, double xEps = highPrecEps,
    int maxEvals = 1000)
{
    int D = BroydenSecant<X>::getD(x0), failCount = 0;
    X x = x0, fx = f(x);
    assert(D == BroydenSecant<X>::getD(fx) && maxEvals >= 2 * D + 1);
    typedef typename BroydenSecant<X>::InverseOperator BIO;
    BIO B(f, x);
    maxEvals -= 2 * D + 1;
    double lastGoodNorm = 1, xError = numeric_limits<double>::infinity();
    if(!isfinite(normInf(fx))) return make_pair(x, xError);
    while(maxEvals > 0)
    {
        assert(normInf(f(x)) <= normInf(f(x0)));
        if(failCount > 1)
        {//after 2nd fail try random step
            --maxEvals;
            assert(normInf(f(x)) <= normInf(f(x0)));
            if(!equationTryRandomStep(f, x, fx, lastGoodNorm))
            {
                assert(normInf(f(x)) <= normInf(f(x0)));
                if(maxEvals >= 2 * D + 1)//need enough evals for next step
                {
                    B = BIO(f, x);
                    maxEvals -= 2 * D;
                    failCount = 0;//back to normal
                }//else keep making random steps
                continue;
            }
            assert(normInf(f(x)) <= normInf(f(x0)));
        }
        X dx = B * -fx, oldFx = fx, oldX = x;
        double ndx = normInf(dx);
        if(!isfinite(ndx))
        {//probably singular, either after bad update or reestimation
            ++failCount;
            continue;
        }
        if(ndx < xEps * (1 + normInf(x))) break;//full step too small
        if(!equationBacktrack(f, x, fx, maxEvals, dx, xEps))
        {
            assert(normInf(f(x)) <= normInf(f(x0)));
            xError = lastGoodNorm = ndx;//last successful full step
            failCount = 0;
        }
        else ++failCount;
            assert(normInf(f(x)) <= normInf(f(x0)));
        if(failCount == 1)//after first fail reestimate J
            if(maxEvals >= 2 * D + 1)//need enough evals for next step
            {
                B = BIO(f, x);
                maxEvals -= 2 * D;
            }
            else ++failCount;//if cant do steps
        else B.addUpdate(fx - oldFx, x - oldX);
        assert(normInf(f(x)) <= normInf(f(x0)));
    }
    return make_pair(x, xError);
}//for type safety  such such int x0 use wrapper
template<typename FUNCTION> pair<double, double> solveSecant(FUNCTION const&
    f, double const& x0, double xEps = highPrecEps, int maxEvals = 1000)
    {return solveBroyden(f, x0, xEps, maxEvals);}

class BroydenLMInverseOperator
{
    typedef Vector<double> X;
    int m;
    Queue<pair<X, X> > updates;
    struct Result
    {
        X Bfx;
        Vector<X> Bdfs, dxBs;
    };
public:
    BroydenLMInverseOperator(int theM): m(theM){}
    void addUpdate(X const& df, X const& dx)
    {
        if(updates.getSize() == m) updates.pop();
        updates.push(make_pair(df, dx));
    }
    X operator*(X const& fx)const
    {
        int n = updates.getSize();
        X Bfx = fx;//base case identity
        Vector<X> Bdfs(n), dxBs(n);
        for(int i = 0; i < n; ++i)
        {
            Bdfs[i] = updates[i].first;
            dxBs[i] = updates[i].second;
        }
        for(int i = 0; i < n; ++i)
        {
            X u = (updates[i].second - Bdfs[i]) *
                (1/dotProduct(updates[i].second, Bdfs[i]));
            if(!isfinite(normInf(u))) continue;//guard against div by 0
            Bfx += outerProductMultLeft(u, dxBs[i], fx);
            for(int j = i + 1; j < n; ++j)
            {
                Bdfs[j] += outerProductMultLeft(u, dxBs[i],updates[j].first);
                dxBs[j] += outerProductMultRight(u, dxBs[i],
                    updates[j].second);
            }
        }
        return Bfx;
    }
};
template<typename FUNCTION> pair<Vector<double>, double> solveLMBroyden(
    FUNCTION const& f, Vector<double> const& x0, double xEps = highPrecEps,
    int maxEvals = 1000, int m = 30)
{
    int D = x0.getSize();
    BroydenLMInverseOperator B(m);
    Vector<double> x = x0, fx = f(x), xBest = x, fBest = fx;
    double s = 1, xError = numeric_limits<double>::infinity(),eBest = xError;
    while(maxEvals-- > 0)
    {
        Vector<double> dx = B * -fx * s;
        double ndx = normInf(dx)/s;//norm of full step
        //something wrong or step too small
        if(!isfinite(ndx) || ndx < xEps * (1 + normInf(x))) break;
        Vector<double> fNew = f(x + dx);
        if(normInf(fNew) > 10 * normInf(fx)) s /= 10;
        else
        {
            B.addUpdate(fNew - fx, dx);
            fx = fNew;
            x += dx;
            xError = ndx;//use last full step
            s = 1;//back to normal step
            if(normInf(fx) < normInf(fBest))
            {
                xBest = x;
                fBest = fx;
                eBest = xError;
            }
        }
    }
    return make_pair(xBest, eBest);
}

template<typename FUNCTION> pair<Vector<double>, double> solveBroydenHybrid(
    FUNCTION const& f, Vector<double> const& x0, double xEps = highPrecEps,
    int maxEvals = 1000, int changeD = 200)
{
    return x0.getSize() > changeD ? solveLMBroyden(f, x0, xEps, maxEvals) :
        solveBroyden(f, x0, xEps, maxEvals);
}

template<typename FUNCTION> pair<double, double> solveSecantGlobal(
    FUNCTION const& f, double xEps = highPrecEps, int nSamples = 100)
{
    assert(nSamples > 0 && xEps >= numeric_limits<double>::epsilon());
    pair<double, double> best;
    double nBestfx;
    for(int i = 0; i < nSamples; ++i)
    {
        double x = GlobalRNG().Levy() * GlobalRNG().sign();
        pair<double, double> next = solveBroyden(f, x, xEps);
        double nNextfx = normInf(f(next.first));
        if(i == 0 || nNextfx < nBestfx)
        {
            best = next;
            nBestfx = nNextfx;
        }
    }
    return best;
}

template<typename FUNCTION> pair<Vector<double>, double> solveBroydenLevy(
    FUNCTION const& f, int D, double xEps = highPrecEps, int nSamples = 1000)
{
    assert(nSamples > 0 && xEps >= numeric_limits<double>::epsilon());
    typedef Vector<double> X;
    pair<X, double> best;
    double nBestfx;
    for(int i = 0; i < nSamples; ++i)
    {
        X x = GlobalRNG().randomUnitVector(D) * GlobalRNG().Levy();
        pair<X, double> next = solveBroydenHybrid(f, x, xEps);
        double nNextfx = normInf(f(next.first));
        if(i == 0 || nNextfx < nBestfx)
        {
            best = next;
            nBestfx = nNextfx;
        }
    }
    return best;
}

template<typename FUNCTION> Vector<double> findAllRealRootsCheb(
    FUNCTION const& f, double a, double b, int maxDegree = 32,
    double duplicateXEps = highPrecEps)
{
    Vector<pair<pair<double, double>, ScaledChebAB> > pieces =
        interpolateAdaptiveHeap<IntervalCheb>(f, a, b, maxDegree).first.
        getPieces();
    PiecewiseData<EMPTY> resultFilter(duplicateXEps);
    for(int i = 0; i < pieces.getSize(); ++i)
    {
        Vector<double> rootsI = pieces[i].second.findAllRealRoots();
        for(int j = 0; j < rootsI.getSize(); ++j)
        {//range and finiteness check
            double polishedRoot =
                solveSecant(f, rootsI[j], duplicateXEps).first;
            if(isfinite(polishedRoot) && a <= polishedRoot &&
                polishedRoot <= b) resultFilter.insert(polishedRoot, EMPTY());
        }
    }
    Vector<pair<double, EMPTY> > tempResult = resultFilter.getPieces();
    Vector<double> result(tempResult.getSize());
    for(int i = 0; i < tempResult.getSize(); ++i)
        result[i] = tempResult[i].first;
    return result;
}

template<typename YX_FUNCTION>
Vector<double> evalYX(double x, Vector<double> const& y, YX_FUNCTION const& f)
{//assume last arg is x
    Vector<double> yAugmented = y;
    yAugmented.append(x);
    Vector<double> fAugmented = f(yAugmented);
    fAugmented.removeLast();
    return fAugmented;
}
template<typename YX_FUNCTION> struct RadauIIA5Function
{
    Vector<double> y;
    double x, h;
    YX_FUNCTION f;
    Vector<double> operator()(Vector<double> fSumDiffs)const
    {
        assert(fSumDiffs.getSize() % 3 == 0);
        int D = fSumDiffs.getSize()/3;
        double s6 = sqrt(6), ci[3] = {(4 - s6)/10, (4 + s6)/10, 1}, A[3][3] =
        {
            {(88 - 7 * s6)/360, (296 - 169 * s6)/1800, (-2 + 3 * s6)/225},
            {(296 + 169 * s6)/1800, (88 + 7 * s6)/360, (-2 - 3 * s6)/225},
            {(16 - s6)/36, (16 + s6)/36, 1.0/9}
        };
        Vector<Vector<double> > ki(3);
        for(int i = 0; i < 3; ++i)
        {
            Vector<double> yi(y);
            for(int j = 0; j < D; ++j) yi[j] += h * fSumDiffs[i * D + j];
            ki[i] = evalYX(x + h * ci[i], yi, f);
        }
        for(int i = 0; i < 3; ++i)
        {
            Vector<double> fSumi(D, 0);
            for(int j = 0; j < 3; ++j) fSumi += ki[j] * A[i][j];
            for(int j = 0; j < D; ++j)//convert fixed point to 0
                fSumDiffs[i * D + j] = fSumi[j] - fSumDiffs[i * D + j];
        }
        return fSumDiffs;
    }
};
struct RadauIIA5StepF
{
    template<typename YX_FUNCTION> Vector<double> operator()(
        YX_FUNCTION const& f, double x, Vector<double> y, double h,
        double solveERelAbs)const
    {
        RadauIIA5Function<YX_FUNCTION> r5f = {y, x, h, f};
        int D = y.getSize();
        Vector<double> fSumDiffs(3 * D, 0);
        fSumDiffs = solveBroyden(r5f, fSumDiffs, max(solveERelAbs,
            numeric_limits<double>::epsilon() * normInf(y)/h)).first;
        for(int j = 0; j < D; ++j) y[j] += h * fSumDiffs[2 * D + j];
        return y;
    }
};

template<typename YX_FUNCTION, typename STEPF> pair<Vector<double>, double>
    adaptiveStepper(YX_FUNCTION const& f, STEPF const& s, double x0,
    double xGoal, Vector<double> y0, double localERelAbs = defaultPrecEps,
    int maxIntervals = 100000, int minIntervals = -1,
    double upFactor = pow(2, 0.2))
{//assume no reuse of f0
    if(minIntervals == -1) minIntervals = sqrt(maxIntervals);
    assert(xGoal > x0 && minIntervals > 0 && upFactor > 1);
    int D = y0.getSize();
    double hMax = (xGoal - x0)/minIntervals, hMin = (xGoal - x0)/maxIntervals,
        linearError = 0, h1 = hMax;
    Vector<double> y = y0,
        y1 = Vector<double>(D, numeric_limits<double>::quiet_NaN()), f0;
    bool last = false;
    for(double x = x0; !last;)
    {
        if(x + h1 > xGoal)
        {//make last step accurate
            h1 = xGoal - x;
            last = true;
        }
        double h2 = h1/2, xFraction = h1/(xGoal - x0),
            tolERelAbs = max(highPrecEps, localERelAbs * sqrt(xFraction)),
            solveERelAbs = tolERelAbs/10;
        if(isnan(normInf(y1))) y1 = s(f, x, y, h1, solveERelAbs);
        Vector<double> y2 = s(f, x, y, h2, solveERelAbs),
            firstY2 = y2;
        y2 = s(f, x + h2, y2, h2, solveERelAbs);
        double normError = normInf(y2 - y1), normY2 = normInf(y2);
        if(h2 < hMin || isEEqual(normY2 + normError, normY2, tolERelAbs))
        {//accept step
            x += h1;
            y = y2;
            linearError += normError;
            y1 = Vector<double>(D, numeric_limits<double>::quiet_NaN());
            if(h2 >= hMin) h1 = min(hMax, h1 * upFactor);//use larger step
        }
        else
        {//use half step
            y1 = firstY2;
            h1 = h2;
            last = false;
        }
    }
    return make_pair(y, linearError);
}
template<typename YX_FUNCTION, typename BOUNDARY_FUNCTION>
struct BoundaryFunctor
{
    YX_FUNCTION const& f;
    BOUNDARY_FUNCTION const& bf;
    double x0, xGoal;
    double operator()(double b)const
    {
        return bf.evaluateGoal(adaptiveStepper(f, RadauIIA5StepF(),
            x0, xGoal, bf.getInitial(b)).first);
    }
};
template<typename YX_FUNCTION, typename BOUNDARY_FUNCTION>
    Vector<Vector<double> >solveBoundaryValue(YX_FUNCTION const& f, double x0,
    double xGoal, Vector<double> const& xPoints, BOUNDARY_FUNCTION const& bf,
    double b0 = 0)
{
    BoundaryFunctor<YX_FUNCTION, BOUNDARY_FUNCTION> fu = {f, bf, x0, xGoal};
    double bFound = solveSecant(fu, b0).first;
    Vector<Vector<double> > result;
    if(isfinite(bFound))
    {
        Vector<double> y0 = bf.getInitial(bFound);
        for(int i = 0; i < xPoints.getSize(); ++i)
        {
            y0 = adaptiveStepper(f, RadauIIA5StepF(), x0, xPoints[i],
                y0).first;
            x0 = xPoints[i];
            result.append(y0);
        }
    }
    return result;
}

struct Cheb1DIntegrator
{//ensure getting error estimate
    template<typename FUNCTION> pair<double, double> operator()(
        FUNCTION const& f, double a, double b, int maxEvals)
        {return integrateCC(f, a, b, maxEvals, min(17, (maxEvals - 1)/2 + 1));}
};
template<typename FUNCTION, typename INTEGRATOR1D = Cheb1DIntegrator>
class RecursiveIntegralFunction
{
    FUNCTION f;
    mutable Vector<double> xBound;
    typedef Vector<pair<double, double> > BOX;
    BOX box;
    mutable Vector<pair<double, long long> >* errors;//copy-proof
    int maxEvalsPerDim;//default for low D power of 2 + 1
    pair<double, double> integrateHelper()const
    {
        INTEGRATOR1D i1D;
        int i = xBound.getSize();
        return i1D(*this, box[i].first, box[i].second, maxEvalsPerDim);
    }
public:
    RecursiveIntegralFunction(BOX const& theBox, int theMaxEvalsPerDim = 33,
        FUNCTION const& theF = FUNCTION()): f(theF), box(theBox), errors(0),
        maxEvalsPerDim(theMaxEvalsPerDim){}
    pair<double, double> integrate()const//the main function
    {
        errors = new Vector<pair<double, long long> >(box.getSize());
        pair<double, double> result = integrateHelper();
        double error = 0;
        for(int i = box.getSize() - 1; i >= 0; --i) error = (box[i].second -
            box[i].first) * (error + (*errors)[i].first/(*errors)[i].second);
        result.second += error;
        delete errors;
        return result;
    }
    double operator()(double x)const//called by INTEGRATOR1D
    {
        xBound.append(x);
        bool recurse = xBound.getSize() < box.getSize();
        pair<double, double> result = recurse ? integrateHelper() :
            make_pair(f(xBound), numeric_limits<double>::epsilon());
        if(!recurse) result.second *= max(1.0, abs(result.first));
        xBound.removeLast();
        int i = xBound.getSize();
        (*errors)[i].first += result.second;
        ++(*errors)[i].second;
        return result.first;
    }
};


}
#endif
