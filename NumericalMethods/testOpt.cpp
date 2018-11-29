#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <limits>
#include <memory> //for shared ptr
#include "NumericalOptimization.h"
#include "../RandomNumberGeneration/Statistics.h"
#include "../Utils/DEBUG.h"
#include "../ExternalMemoryAlgorithms/CSV.h"
using namespace std;
using namespace igmdk;

struct TestFunctionsMin
{
    struct BaseF
    {
        virtual Vector<double> operator()(Vector<double> const& x)const = 0;
        virtual string name()const = 0;
        virtual Vector<double> getX0()const = 0;
        virtual Vector<double> getAnswer()const = 0;
    };
    struct ExtendedRosenbrock: public BaseF
    {//From Dennis & Schnabel
        int n;
        ExtendedRosenbrock(int theN = 2): n(theN) {assert(theN % 2 == 0);}
        Vector<double> operator()(Vector<double> const& x)const
        {
            Vector<double> fx(n);
            for(int i = 0; i < n/2; ++i)
            {
                double i1 = 2 * i, i2 = i1 + 1;
                fx[i1] = 10 * (x[i2] - x[i1] * x[i1]);
                fx[i2] = 1 - x[i1];
            }
            return fx;
        }
        string name()const{return "ExtendedRosenbrock" + toString(n);}
        Vector<double> getX0()const
        {
            Vector<double> x0 = Vector<double>(n, 1);
            for(int i = 0; i < n/2; ++i) x0[2 * i] = 1.2;
            return x0;
        }
        Vector<double> getAnswer()const{return Vector<double>(n, 1);}
    };
    struct ExtendedPowellSingular: public BaseF
    {//From Dennis & Schnabel, J singular at solution
        int n;
        ExtendedPowellSingular(int theN = 4): n(theN) {assert(theN % 4 == 0);}
        Vector<double> operator()(Vector<double> const& x)const
        {
            Vector<double> fx(n);
            for(int i = 0; i < n/4; ++i)
            {
                double i1 = 4 * i, i2 = i1 + 1, i3 = i2 + 1, i4 = i3 + 1;
                fx[i1] = x[i1] + 10 * x[i2];
                fx[i2] = sqrt(5) * (x[i3] - x[i4]);
                fx[i3] = (x[i2] - 2 * x[i3]) * (x[i2] - 2 * x[i3]);
                fx[i4] = sqrt(10) * (x[i1] - x[i4]) * (x[i1] - x[i4]);
            }
            return fx;
        }
        string name()const{return "ExtendedPowellSingular" + toString(n);}
        Vector<double> getX0()const
        {
            Vector<double> x0 = Vector<double>(n, 1);
            for(int i = 0; i < n/4; ++i)
            {
                x0[4 * i] = 3;
                x0[4 * i + 1] = -1;
                x0[4 * i + 2] = 0;
                x0[4 * i + 2] = 1;
            }
            return x0;
        }
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct HelicalValley: public BaseF
    {//From Dennis & Schnabel
        Vector<double> operator()(Vector<double> const& x)const
        {
            Vector<double> fx(3);
            double q = 0.5/PI() * atan(x[1]/x[0]);
            if(x[0] < 0) q += 0.5;
            fx[0] = 10 * (x[2] - 10 * q);
            fx[1] = 10 * (sqrt(x[0] * x[0] + x[1] * x[1]) - 1);
            fx[2] = x[2];
            return fx;
        }
        string name()const{return "HelicalValley";}
        Vector<double> getX0()const
        {
            Vector<double> x0(3, 0);
            x0[0] = -1;
            return x0;
        }
        Vector<double> getAnswer()const{return -getX0();}
    };
    struct VariableDimensionF: public BaseF
    {//From More et al
        int n;
        VariableDimensionF(int theN = 2): n(theN) {}
        Vector<double> operator()(Vector<double> const& x)const
        {
            Vector<double> fx(n + 2);
            for(int i = 0; i < n; ++i)
            {
                fx[i] = x[i] - 1;
                fx[n] += (i + 1) * fx[i];
            }
            fx[n + 1] = fx[n] * fx[n];
            return fx;
        }
        string name()const{return "VariableDimensionF" + toString(n);}
        Vector<double> getX0()const
        {
            Vector<double> x0(n, 1);
            for(int i = 0; i < n; ++i) x0[i] -= (i + 1.0)/n;
            return x0;
        }
        Vector<double> getAnswer()const{return Vector<double>(n, 1);}
    };
    struct LinearFFullRank: public BaseF
    {//From More et al
        int n;
        LinearFFullRank(int theN = 2): n(theN) {}
        Vector<double> operator()(Vector<double> const& x)const
        {
            double sum = 0;
            for(int i = 0; i < n; ++i) sum += x[i];
            Vector<double> fx(n);
            for(int i = 0; i < n; ++i) fx[i] = x[i] - 2 * sum/n - 1;
            return fx;
        }
        string name()const{return "LinearFFullRank" + toString(n);}
        Vector<double> getX0()const{return Vector<double>(n, 1);}
        Vector<double> getAnswer()const{return -Vector<double>(n, 1);}
    };
    struct BrownBadScaled: public BaseF
    {//From More et al
        Vector<double> operator()(Vector<double> const& x)const
        {
            Vector<double> fx(3);
            fx[0] = x[0] - 1000000;
            fx[1] = x[1] - 2.0/1000000;
            fx[2] = x[0] * x[1] - 2;
            return fx;
        }
        string name()const{return "BrownBadScaled";}
        Vector<double> getX0()const{return Vector<double>(2, 1);}
        Vector<double> getAnswer()const
        {
            Vector<double> x0(2, 0);
            x0[0] = 1000000;
            x0[1] = 2.0/1000000;
            return x0;
        }
    };
    struct Beale: public BaseF
    {//From More et al
        Vector<double> operator()(Vector<double> const& x)const
        {
            Vector<double> fx(3);
            fx[0] = 1.5;
            fx[1] = 2.25;
            fx[2] = 2.625;
            for(int i = 0; i < fx.getSize(); ++i)
                fx[i] -= x[0] * (1 - pow(x[1], i + 1));
            return fx;
        }
        string name()const{return "Beale";}
        Vector<double> getX0()const{return Vector<double>(2, 1);}
        Vector<double> getAnswer()const
        {
            Vector<double> x0(2, 0);
            x0[0] = 3;
            x0[1] = 0.5;
            return x0;
        }
    };
    struct BiggsExp6: public BaseF
    {//From More et al;
        Vector<double> operator()(Vector<double> const& x)const
        {
            Vector<double> fx(6);
            for(int i = 0; i < fx.getSize(); ++i)
            {
                double ti = (i + 1.0)/10,
                    yi = exp(-ti) - 5 * exp(-10 * ti) + 3 * exp(-4 * ti);
                fx[i] = x[2] * exp(-ti * x[0]) - x[3] * exp(-ti * x[1]) +
                    x[5] * exp(-ti * x[4]) - yi;
            }
            return fx;
        }
        string name()const{return "BiggsExp6";}
        Vector<double> getX0()const
        {
            Vector<double> x0(6, 1);
            x0[1] = 2;
            return x0;
        }
        Vector<double> getAnswer()const
        {
            Vector<double> a(6);
            a[0] = 1;
            a[1] = 10;
            a[2] = 1;
            a[3] = 5;
            a[4] = 4;
            a[5] = 3;
            return a;
        }
    };
    static int evalCount;
    struct MetaF
    {
        shared_ptr<BaseF> f;
        template<typename F> MetaF(shared_ptr<F> const& theF): f(theF){}
        double operator()(Vector<double> const& x)const
        {
            ++evalCount;
            return norm((*f)(x));
        }
        string getName()const{return f->name();}
        Vector<double> getX0()const{return f->getX0();}
        pair<Vector<double>, double> getAnswer()const
            {return make_pair(f->getAnswer(), norm((*f)(f->getAnswer())));}
    };
    static Vector<MetaF> getFunctions()
    {
        Vector<MetaF> result;
        result.append(MetaF(make_shared<ExtendedRosenbrock>()));
        result.append(MetaF(make_shared<ExtendedPowellSingular>()));
        result.append(MetaF(make_shared<HelicalValley>()));
        result.append(MetaF(make_shared<VariableDimensionF>()));
        result.append(MetaF(make_shared<LinearFFullRank>()));
        result.append(MetaF(make_shared<BrownBadScaled>()));
        result.append(MetaF(make_shared<Beale>()));
        result.append(MetaF(make_shared<BiggsExp6>()));
        result.append(MetaF(make_shared<ExtendedRosenbrock>(10)));
        result.append(MetaF(make_shared<ExtendedPowellSingular>(12)));
        result.append(MetaF(make_shared<VariableDimensionF>(10)));
        result.append(MetaF(make_shared<LinearFFullRank>(10)));
        result.append(MetaF(make_shared<ExtendedRosenbrock>(30)));
        result.append(MetaF(make_shared<ExtendedPowellSingular>(32)));
        result.append(MetaF(make_shared<VariableDimensionF>(30)));
        result.append(MetaF(make_shared<LinearFFullRank>(30)));
        result.append(MetaF(make_shared<ExtendedRosenbrock>(100)));
        result.append(MetaF(make_shared<ExtendedPowellSingular>(100)));
        result.append(MetaF(make_shared<VariableDimensionF>(100)));
        result.append(MetaF(make_shared<LinearFFullRank>(100)));
        //large D functions
        result.append(MetaF(make_shared<ExtendedRosenbrock>(1000)));
        result.append(MetaF(make_shared<ExtendedPowellSingular>(1000)));
        result.append(MetaF(make_shared<VariableDimensionF>(1000)));
        result.append(MetaF(make_shared<LinearFFullRank>(1000)));
        result.append(MetaF(make_shared<ExtendedRosenbrock>(10000)));
        result.append(MetaF(make_shared<ExtendedPowellSingular>(10000)));
        result.append(MetaF(make_shared<VariableDimensionF>(10000)));
        result.append(MetaF(make_shared<LinearFFullRank>(10000)));
        return result;
    }
};
int TestFunctionsMin::evalCount = 0;

//USE SAME BACKTRACK FOR EQ AND OPT USING FUNCTION X *X?
template<typename FUNCTION> bool backtrackLineSearch(FUNCTION const& f,
    Vector<double> const& gradient, Vector<double>& x, double& fx,
    int& maxEvals, Vector<double> const& dx, double yEps)
{
    double fxFirst = fx, dd = -(dx * gradient), minDescent = dd * 0.0001;
    if(!isfinite(minDescent) || minDescent <= 0) return true;
    for(double s = 1; maxEvals > 0; s /= 2)
    {
        double fNew = f(x + dx * s), fGoal = fx - minDescent * s;
        --maxEvals;
        if(fGoal > fNew)
        {
            x += dx * s;
            fx = fNew;
            break;
        }//step to small if goal was with c = 1
        else if(!isELess(fxFirst - dd * s, fx, yEps)) break;
    }
    return !isELess(fx, fxFirst, yEps);//must make good progress
}

template<typename FUNCTION, typename GRADIENT, typename DIRECTIONAL_DERIVATIVE>
    pair<Vector<double>, double> conjugateGradient(Vector<double> const& x0,
    FUNCTION const& f, GRADIENT const& g, DIRECTIONAL_DERIVATIVE const& dd,
    int maxEvals = 1000000, double yPrecision = highPrecEps,
    string const& formula = "PRP+", bool useExact = true)
{
    pair<Vector<double>, double> xy(x0, f(x0));
    Vector<double> grad = g(xy.first), d;
    int D = xy.first.getSize(), gEvals = g.fEvals(D),
        restartDelta = 100, restartCountdown = 0;
    maxEvals -= gEvals;
    double a;
    while(maxEvals > 0)
    {
        if(restartCountdown <= 0)
        {
            restartCountdown = restartDelta;
            d = -grad;
            a = steepestStepScale(xy.first, xy.second, grad);
        }
        Vector<double> xOld = xy.first;
        if(goldenSectionLineSearch(f, grad, dd, xy.first, xy.second, maxEvals,
            d * a, yPrecision, useExact))
        {//failed case
            if(restartCountdown == restartDelta) break;//failed after restart
            else
            {//force restart
                restartCountdown = 0;
                continue;
            }
        }
        else --restartCountdown;
        if((maxEvals -= gEvals) < 1) break;
        Vector<double> gradNew = g(xy.first);
        double b = 0;
        if(formula == "PRP+")
            b = max(0.0, dotProduct(gradNew, (gradNew - grad))/dotProduct(grad, grad));
        else if(formula == "HZ")
        {
            Vector<double> y = gradNew - grad;
            double temp = dotProduct(y, d),
                b = dotProduct(y - d * (dotProduct(y, y) * 2/temp), gradNew)/temp;
        }
        else
        {
            DEBUG("bad formula specification");
            assert(false);
        }
        double temp = dotProduct(grad, (xy.first - xOld));
        d = -gradNew + d * b;
        grad = gradNew;
        a = dotProduct(grad, d)/temp;
    }
    return xy;
}

template<typename FUNCTION, typename GRADIENT, typename DIRECTIONAL_DERIVATIVE>
    pair<Vector<double>, double> LBFGSMinimizeNW(Vector<double> const& x0,
    FUNCTION const& f, GRADIENT const& g, DIRECTIONAL_DERIVATIVE const& dd,
    int maxEvals = 1000000, double yPrecision = highPrecEps,
    int historySize = 8, bool useExact = true)
{
    typedef Vector<double> V;
    Queue<pair<V, V> > history;
    pair<V, double> xy(x0, f(x0));
    V grad = g(xy.first), d = -grad;
    int D = xy.first.getSize(), gEvals = g.fEvals(D);
    maxEvals -= 1 + gEvals;
    while(maxEvals > 0)
    {//backtrack using d to get sufficient descent
        pair<V, double> xyOld = xy;
        bool failed = goldenSectionLineSearch(f, grad, dd, xy.first, xy.second,
            maxEvals, d, yPrecision, useExact);
        if(failed || (maxEvals -= gEvals) <= 0) break;
        V newGrad = g(xy.first);
        if(history.getSize() >= historySize) history.pop();
        history.push(make_pair(xy.first - xyOld.first, newGrad - grad));
        //"double recursion" algorithm to update d
        d = grad = newGrad;
        Vector<double> a, p;
        int last = history.getSize() - 1;
        for(int i = last; i >= 0; --i)
        {
            double pi = 1/dotProduct(history[i].first, history[i].second),
                ai = dotProduct(history[i].first, d) * pi;
            d -= history[i].second * ai;
            a.append(ai);
            p.append(pi);
        }//initial Hessian is scaled diagonal
        d *= 1/(dotProduct(history[last].second, history[last].second) * p[last]);
        for(int i = 0; i < history.getSize(); ++i)
        {
            double bi = dotProduct(history[i].second, d) * p[last - i];
            d += history[i].first * (a[last - i] - bi);
        }
        d *= -1;
    }
    return xy;
}

template<typename FUNCTION, typename GRADIENT, typename DIRECTIONAL_DERIVATIVE>
    pair<Vector<double>, double> LBFGSMinimizeScale(Vector<double> const& x0,
    FUNCTION const& f, GRADIENT const& g, DIRECTIONAL_DERIVATIVE const& dd,
    string scale, int maxEvals = 1000000, double yPrecision = highPrecEps,
    int historySize = 8, bool useExact = true)
{
    typedef Vector<double> V;
    Queue<pair<V, V> > history;
    pair<V, double> xy(x0, f(x0));
    V grad = g(xy.first), d;
    int D = xy.first.getSize(), gEvals = g.fEvals(D),
        restartDelta = 100, restartCountdown = 0;
    maxEvals -= 1 + gEvals;
    while(maxEvals > 0)
    {//backtrack using d to get sufficient descent
        if(restartCountdown <= 0)
        {
            restartCountdown = restartDelta;
            if(scale == "No Scale") d = -grad;
            else if(scale == "DS") d = grad * (-1/max(1.0, abs(xy.second)));
            else if(scale == "X Scale") d = grad * (-max(1.0, norm(xy.first))/10/norm(grad));
            else assert(false);
            history = Queue<pair<V, V> >();
        }
        pair<V, double> xyOld = xy;
        if(goldenSectionLineSearch(f, grad, dd, xy.first, xy.second, maxEvals, d,
            yPrecision, useExact))
        {//failed case
            if(restartCountdown == restartDelta) break;//failed after restart
            else
            {//force restart
                restartCountdown = 0;
                continue;
            }
        }
        else --restartCountdown;
        if((maxEvals -= gEvals) < 1) break;
        V newGrad = g(xy.first);
        if(history.getSize() >= historySize) history.pop();
        history.push(make_pair(xy.first - xyOld.first, newGrad - grad));
        //"double recursion" algorithm to update d
        d = grad = newGrad;
        Vector<double> a, p;
        int last = history.getSize() - 1;
        for(int i = last; i >= 0; --i)
        {
            double pi = 1/dotProduct(history[i].first, history[i].second),
                ai = dotProduct(history[i].first, d) * pi;
            d -= history[i].second * ai;
            a.append(ai);
            p.append(pi);
        }//initial Hessian is scaled diagonal
        d *= 1/(dotProduct(history[last].second, history[last].second) * p[last]);
        for(int i = 0; i < history.getSize(); ++i)
        {
            double bi = dotProduct(history[i].second, d) * p[last - i];
            d += history[i].first * (a[last - i] - bi);
        }
        d *= -1;
    }
    return xy;
}

template<typename FUNCTION, typename GRADIENT, typename DIRECTIONAL_DERIVATIVE>
    pair<Vector<double>, double> BFGSMinimize(
    Vector<double> const& x0, FUNCTION const& f,
    GRADIENT const& g, DIRECTIONAL_DERIVATIVE const& dd,
    int maxEvals = 1000000,
    double yPrecision = highPrecEps, bool useExact = true)
{
    typedef Vector<double> V;
    int D = x0.getSize(), gEvals = g.fEvals(D),
        restartDelta = 100, restartCountdown = 0;
    pair<V, double> xy(x0, f(x0));
    V grad = g(xy.first);
    Matrix<double> I = Matrix<double>::identity(D), H = I;//dummy init
    maxEvals -= gEvals + 1;
    bool firstStep = true;
    while(maxEvals > 0)
    {//backtrack using d to get sufficient descent
        if(restartCountdown <= 0)
        {
            restartCountdown = restartDelta;
            H = I * (1/steepestStepScale(xy.first, xy.second, grad));
            firstStep = true;
        }
        V d = H * -grad;
        pair<V, double> xyOld = xy;
        if(goldenSectionLineSearch(f, grad, dd, xy.first, xy.second, maxEvals, d,
            yPrecision, useExact))
        {//failed case
            if(restartCountdown == restartDelta) break;//failed after restart
            else
            {//force restart
                restartCountdown = 0;
                continue;
            }
        }
        else --restartCountdown;
        if((maxEvals -= gEvals) < 1) break;
        V newGrad = g(xy.first), s = xy.first - xyOld.first,
            y = newGrad - grad;
        double p = 1/dotProduct(y, s);
        if(firstStep)
        {//Shanno-Phua scaling per Dennis & Schnabel
            H = dotProduct(y, y)/p * I;
            firstStep = false;
        }
        H = (I - p * outerProduct(s, y)) * H * (I - p * outerProduct(y, s)) +
            p * outerProduct(s, s);
        grad = newGrad;
    }
    return xy;
}

template<typename TEST_SET, typename FUNCTION> void debugResultHelper(
    pair<Vector<double>, double> const& result,
    FUNCTION const& f, Vector<Vector<string> > & matrix, int start)
{
    //for(int i = 0; i < result.first.getSize(); ++i) DEBUG(result.first[i]);
    //DEBUG(result.second);
    //DEBUG(f(result.first));
    double timediff = 1.0 * (clock() - start)/CLOCKS_PER_SEC;
    pair<Vector<double>, double> answer = f.getAnswer();
    double eps = numeric_limits<double>::epsilon(),
        relAbsXNorm = max(eps, normInf(result.first - answer.first)/
        max(1.0, normInf(answer.first))),
        relAbsYNorm = max(eps, abs(result.second - answer.second)/
            max(1.0, abs(answer.second)));
    DEBUG(relAbsXNorm);
    DEBUG(relAbsYNorm);
    DEBUG(TEST_SET::evalCount);
    matrix.lastItem().append(toString(relAbsXNorm));
    matrix.lastItem().append(toString(relAbsYNorm));
    matrix.lastItem().append(toString(TEST_SET::evalCount));
    TEST_SET::evalCount = 0;
    double gradNorm = norm(estimateGradientCD(result.first, f)),
        normalizedGradNorm = gradNorm/max(1.0, abs(result.second));
    TEST_SET::evalCount = 0;
    DEBUG(normalizedGradNorm);
    matrix.lastItem().append(toString(normalizedGradNorm));
    DEBUG(timediff);
    matrix.lastItem().append(toString(timediff));
}

template<typename POINT, typename FUNCTION> void debugResultNew(
    pair<POINT, double> const& result,
    FUNCTION const& f, Vector<Vector<string> > & matrix, int start)
{
    debugResultHelper<TestFunctionsMin>(result, f, matrix, start);
}

template<typename FUNCTION> void testAllSolversLargeD(FUNCTION const& f,
    Vector<Vector<string> >& matrix)
{
    int start = 0;
    GradientFunctor<FUNCTION> g(f);
    DirectionalDerivativeFunctor<FUNCTION> dd(f);
    DEBUG("Compass");
    matrix.lastItem().append("Compass");
    debugResultNew(compassMinimize(f.getX0(), f), f, matrix, start);
    DEBUG("UnimodalCD");
    matrix.lastItem().append("UnimodalCD");
    debugResultNew(unimodalCoordinateDescentGeneral(f, f.getX0()), f, matrix, start);
    DEBUG("ConjugateGradient");
    matrix.lastItem().append("ConjugateGradient");
    debugResultNew(conjugateGradient(f.getX0(), f, g, dd), f, matrix, start);
    DEBUG("LBFGSMinimize");
    matrix.lastItem().append("LBFGSMinimize");
    debugResultNew(LBFGSMinimize(f.getX0(), f, g, dd), f, matrix, start);
}
//for unimodal 2 works, for powell probably just lucky step choice gives it good result!
template<typename FUNCTION> void testAllSolvers(FUNCTION const& f, Vector<Vector<string> >& matrix)
{
    GradientFunctor<FUNCTION> g(f);
    DirectionalDerivativeFunctor<FUNCTION> dd(f);
    int D = f.getX0().getSize(), start = 0;
    DEBUG("metaSPSA");
    matrix.lastItem().append("metaSPSA");
    start = clock();
    debugResultNew(metaSPSA(f.getX0(), f), f, matrix, start);
    DEBUG("Compass");
    matrix.lastItem().append("Compass");
    debugResultNew(compassMinimize(f.getX0(), f), f, matrix, start);
    DEBUG("UnimodalCD");
    matrix.lastItem().append("UnimodalCD");
    start = clock();
    debugResultNew(unimodalCoordinateDescentGeneral(f, f.getX0()), f, matrix, start);
    DEBUG("NelderMead");
    matrix.lastItem().append("NelderMead");
    NelderMead<FUNCTION> nm(D, f);
    start = clock();
    debugResultNew(nm.minimize(f.getX0()), f, matrix, start);
    DEBUG("RestartedNelderMead");
    matrix.lastItem().append("RestartedNelderMead");
    NelderMead<FUNCTION> nmr(D, f);
    start = clock();
    debugResultNew(nmr.restartedMinimize(f.getX0()), f, matrix, start);
    DEBUG("RestartedNelderMead100");
    matrix.lastItem().append("RestartedNelderMead100");
    NelderMead<FUNCTION> nmr100(D, f);
    start = clock();
    debugResultNew(nmr100.restartedMinimize(f.getX0(), 10000, highPrecEps, 100), f, matrix, start);

    DEBUG("GradDescentMT");
    matrix.lastItem().append("GradDescentMT");
    start = clock();
    debugResultNew(gradDescent(f.getX0(), f, g, dd, 1000000, highPrecEps, false), f, matrix, start);
    DEBUG("GradDescent");
    matrix.lastItem().append("GradDescent");
    start = clock();
    debugResultNew(gradDescent(f.getX0(), f, g, dd), f, matrix, start);
    DEBUG("SubgradientDescent");
    matrix.lastItem().append("SubgradientDescent");
    start = clock();
    debugResultNew(subgradientDescent(f.getX0(), f, g), f, matrix, start);
    DEBUG("SubgradientDescent2");
    matrix.lastItem().append("SubgradientDescent2");
    start = clock();
    debugResultNew(subgradientDescent2(f.getX0(), f, g), f, matrix, start);

    DEBUG("ConjugateGradientPRP+");
    matrix.lastItem().append("ConjugateGradientPRP+");
    start = clock();
    debugResultNew(conjugateGradient(f.getX0(), f, g, dd), f, matrix, start);
    DEBUG("ConjugateGradientHZ");
    matrix.lastItem().append("ConjugateGradientHZ");
    start = clock();
    debugResultNew(conjugateGradient(f.getX0(), f, g, dd, 1000000, highPrecEps, "HZ"), f, matrix, start);
    DEBUG("LBFGSMinimizeNW");
    matrix.lastItem().append("LBFGSMinimizeNW");
    start = clock();
    debugResultNew(LBFGSMinimizeNW(f.getX0(), f, g, dd), f, matrix, start);
    DEBUG("LBFGSMinimize");
    matrix.lastItem().append("LBFGSMinimize");
    start = clock();
    debugResultNew(LBFGSMinimize(f.getX0(), f, g, dd), f, matrix, start);
    DEBUG("LBFGSMinimizeXScale");
    matrix.lastItem().append("LBFGSMinimizeXScale");
    start = clock();
    debugResultNew(LBFGSMinimizeScale(f.getX0(), f, g, dd, "X Scale"), f, matrix, start);
    DEBUG("LBFGSMinimizeNoScale");
    matrix.lastItem().append("LBFGSMinimizeNoScale");
    start = clock();
    debugResultNew(LBFGSMinimizeScale(f.getX0(), f, g, dd, "No Scale"), f, matrix, start);
    DEBUG("LBFGSMinimizeDS");
    matrix.lastItem().append("LBFGSMinimizeDS");
    start = clock();
    debugResultNew(LBFGSMinimizeScale(f.getX0(), f, g, dd, "DS"), f, matrix, start);
    DEBUG("LBFGSMinimizeMT");
    matrix.lastItem().append("LBFGSMinimizeMT");
    start = clock();
    debugResultNew(LBFGSMinimize(f.getX0(), f, g, dd, 1000000, highPrecEps, 8, false), f, matrix, start);
    DEBUG("BFGSMinimize");
    matrix.lastItem().append("BFGSMinimize");
    start = clock();
    debugResultNew(BFGSMinimize(f.getX0(), f, g, dd), f, matrix, start);
    DEBUG("HybridLocalMinimize");
    matrix.lastItem().append("HybridLocalMinimize");
    start = clock();
    debugResultNew(hybridLocalMinimize(f.getX0(), f), f, matrix, start);
}


struct OptTestFunctions1D
{
    struct BaseF
    {
        virtual double operator()(double const& x)const = 0;
        virtual string name()const = 0;
        virtual double getX0()const{return 0;}
        virtual double getAnswer(){return 3;}
    };
    struct Linear: public BaseF
    {//easy
        double operator()(double const& x)const{return x - 3;}
        string name()const{return "Linear";}
    };
    struct QuadE: public BaseF
    {//hard, differentiable
        double operator()(double const& x)const
        {
            double temp = x - 3;
            return temp * temp - 0.001;
        }
        string name()const{return "QuadE";}
    };
    struct Poly6: public BaseF
    {//differentiable, ill-conditioned
        double operator()(double const& x)const{return pow(x - 3, 6);}
        string name()const{return "Poly6";}
    };
    struct Sqrt2F: public BaseF
    {//differentiable, ill-conditioned
        double a, x0;
        Sqrt2F(double theA = 2, double theX0 = 0): a(theA), x0(theX0){}
        double operator()(double const& x)const{return x * x - a;}
        string name()const{return "Sqrt2F_" + toString(a) + "_" + toString(x0);}
        double getX0()const{return 0;}
        double getAnswer()const{return sqrt(x0);}
    };
    static int evalCount;
    struct MetaF
    {
        shared_ptr<BaseF> f;
        template<typename F> MetaF(shared_ptr<F> const& theF): f(theF){}
        double operator()(double const& x)const
        {
            ++evalCount;
            double temp = (*f)(x);
            return temp * temp;
        }
        string getName()const{return f->name();}
        double getX0()const{return f->getX0();}
        double getAnswer()const{return f->getAnswer();}
    };
    static Vector<MetaF> getFunctions()
    {
        Vector<MetaF> result;
        result.append(MetaF(make_shared<Linear>()));
        result.append(MetaF(make_shared<QuadE>()));
        result.append(MetaF(make_shared<Poly6>()));
        result.append(MetaF(make_shared<Sqrt2F>()));
        result.append(MetaF(make_shared<Sqrt2F>(2, 2)));
        return result;
    }
};
int OptTestFunctions1D::evalCount = 0;

void debugOpt1DResult(pair<double, double> const& result, double answer,
    Vector<Vector<string> >& matrix)
{
    DEBUG(normInf(result.first - answer));
    DEBUG(result.second);
    DEBUG(OptTestFunctions1D::evalCount);
    matrix.lastItem().append(toString(normInf(result.first - answer)));
    matrix.lastItem().append(toString(result.second));
    matrix.lastItem().append(toString(OptTestFunctions1D::evalCount));
    OptTestFunctions1D::evalCount = 0;
}
template<typename FUNCTION> void testOptHelper1D(FUNCTION const& f, double x0,
    Vector<Vector<string> >& matrix)
{
    pair<double, double> result = compassMinimize1D(x0, f);
    DEBUG("compass1D");
    matrix.lastItem().append("compass1D");
    debugOpt1DResult(compassMinimize1D(x0, f), f.getAnswer(), matrix);
    /*DEBUG("GradientDescentDBT1D");
    matrix.lastItem().append("GradientDescentDBT1D");
    debugOpt1DResult(gradientDescentDBT1D(x0, f, DerivFunctor<FUNCTION>(f)), f.getAnswer(), matrix);*/
    DEBUG("BracketGSUnimodal");
    matrix.lastItem().append("BracketGSUnimodal");
    debugOpt1DResult(minimizeGSBracket(f, x0), f.getAnswer(), matrix);
}

void testOpt1D()
{
    Vector<Vector<string> > matrix;
    Vector<OptTestFunctions1D::MetaF> fs = OptTestFunctions1D::getFunctions();
    for(int i = 0; i < fs.getSize(); ++i)
    {
        string name = fs[i].getName();
        DEBUG(name);
        matrix.append(Vector<string>());
        matrix.lastItem().append(name);
        testOptHelper1D(fs[i], fs[i].getX0(), matrix);
    }
    int reportNumber = time(0);
    string filename = "reportOpt1D" + toString(reportNumber) + ".csv";
    createCSV(matrix, filename.c_str());
}


void createMinReport(string const& prefix,
    Vector<Vector<string> > const& matrix, int nRepeats)
{
    int reportNumber = time(0);
    string filename = prefix + toString(reportNumber) + ".csv";
    createCSV(matrix, filename.c_str());
    Vector<string> names;
    names.append("XError");
    names.append("YError");
    names.append("NEvals");
    names.append("ScaledGradNorm");
    names.append("TimeSeconds");
    createAugmentedCSVFiles(matrix, names, filename, nRepeats);
}

void testAllFunctions()
{
    Vector<Vector<string> > matrix;
    string name;
    Vector<TestFunctionsMin::MetaF> fs = TestFunctionsMin::getFunctions();
    for(int i = 0; i < fs.getSize(); ++i)
    {
        string name = fs[i].getName();
        DEBUG(name);
        int D = fs[i].getX0().getSize();
        if(D >= 1000)
        {
            DEBUG("large scale case");
            continue;
        }
        matrix.append(Vector<string>());
        matrix.lastItem().append(name);
        testAllSolvers(fs[i], matrix);
    }
    createMinReport("reportMin", matrix, 1);
}

void testAllFunctionsLargeD()
{
    Vector<Vector<string> > matrix;
    string name;
    Vector<TestFunctionsMin::MetaF> fs = TestFunctionsMin::getFunctions();
    for(int i = 0; i < fs.getSize(); ++i)
    {

        string name = fs[i].getName();
        DEBUG(name);
        int D = fs[i].getX0().getSize();
        if(D < 1000)
        {
            DEBUG("small scale case");
            continue;
        }
        matrix.append(Vector<string>());
        matrix.lastItem().append(name);
        testAllSolversLargeD(fs[i], matrix);
    }
    createMinReport("reportMinLargeD", matrix, 1);
}

struct TestFunctionsGlobalBoxMin
{
    struct BaseF
    {
        virtual double operator()(Vector<double> const& x)const = 0;
        virtual string name()const = 0;
        virtual Vector<pair<double, double> > getBox()const = 0;
        virtual Vector<double> getAnswer()const = 0;
    };
    //separability annotations from Jamil & Yang
    //many visualizations at
    //1. http://infinity77.net/global_optimization/test_functions.html
    //2. https://www.sfu.ca/~ssurjano/optimization.html
    //3. http://al-roomi.org/benchmarks/unconstrained
    struct Ackley: public BaseF
    {//Quadratic with noise; from Simon; non-separable
        int n;
        Ackley(int theN = 2): n(theN) {assert(theN % 2 == 0);}
        double operator()(Vector<double> const& x)const
        {
            double temp = 0;
            for(int i = 0; i < n; ++i)
                temp += cos(2 * PI() * x[i]);
            return 20 + exp(1) -20 * exp(-0.2 * norm(x)) - exp(temp/n);
        }
        string name()const{return "Ackley" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(-30, 30));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct FletcherPowell: public BaseF
    {//Unpredictable landscape; from Simon; non-separable (self-concluded)
        int n;
        Vector<double> a;
        Vector<Vector<double> > aij, bij;
        FletcherPowell(int theN = 2): n(theN), a(n), aij(n, Vector<double>(n)),
            bij(aij)
        {
            assert(n > 1);
            for(int i = 0; i < n; ++i)
            {
                a[i] = GlobalRNG().uniform(-PI(), PI());
                for(int j = 0; j < n; ++j)
                {
                    aij[i][j] = GlobalRNG().uniform(-100, 100);
                    bij[i][j] = GlobalRNG().uniform(-100, 100);
                }
            }
        }
        double operator()(Vector<double> const& x)const
        {
            double sum = 0;
            for(int i = 0; i < n; ++i)
            {
                double Ai = 0, Bi = 0;
                for(int j = 0; j < n; ++j)
                {
                    Ai += aij[i][j] * sin(a[j]) + bij[i][j] * cos(a[j]);
                    Bi += aij[i][j] * sin(x[j]) + bij[i][j] * cos(x[j]);
                }
                sum += (Ai - Bi) * (Ai - Bi);
            }
            return sum;
        }
        string name()const{return "FletcherPowell" + toString(n);}
        Vector<pair<double, double> > getBox()const
        {
            return Vector<pair<double, double> >(n, make_pair(-PI(), PI()));
        }
        Vector<double> getAnswer()const{return a;}
    };
    struct Griewank: public BaseF
    {//Very noisy non-separable; from Simon; non-separable
        int n;
        Griewank(int theN = 2): n(theN) {assert(theN % 2 == 0);}
        double operator()(Vector<double> const& x)const
        {
            double temp = 1 + dotProduct(x, x)/4000, prod = 1;
            for(int i = 0; i < n; ++i) prod *= cos(x[i]/sqrt(i + 1));
            return temp - prod;
        }
        string name()const{return "Griewank" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(-600, 600));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct Rastrigin: public BaseF
    {//Exponentially many in n local minima; from Simon; separable
        int n;
        Rastrigin(int theN = 2): n(theN) {assert(theN % 2 == 0);}
        double operator()(Vector<double> const& x)const
        {
            double temp = 0;
            for(int i = 0; i < n; ++i)
                temp += x[i] * x[i] - 10 * cos(2 * PI() * x[i]);
            return 10 * n + temp;
        }
        string name()const{return "Rastrigin" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(-5.12, 5.12));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct SchwefelDoubleSum: public BaseF
    {//High condition number; from Simon; non-separable
        int n;
        SchwefelDoubleSum(int theN = 2): n(theN) {assert(theN % 2 == 0);}
        double operator()(Vector<double> const& x)const
        {
            double sum = 0;
            for(int i = 0; i < n; ++i)
            {
                double sum2 = 0;
                for(int j = 0; j < i; ++j) sum2 += x[j];
                sum += sum2 * sum2;
            }
            return sum;
        }
        string name()const{return "SchwefelDoubleSum" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(-65.356, 65.356));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct SchwefelMax: public BaseF
    {//Only largest value matters; from Simon; separable
        int n;
        SchwefelMax(int theN = 2): n(theN) {assert(theN % 2 == 0);}
        double operator()(Vector<double> const& x)const{return normInf(x);}
        string name()const{return "SchwefelMax" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(-100, 100));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct StepFunction: public BaseF
    {//Quadratic with plateaus; from Simon; separable
        int n;
        StepFunction(int theN = 2): n(theN) {assert(theN % 2 == 0);}
        double operator()(Vector<double> const& x)const
        {
            double sum = 0;
            for(int i = 0; i < n; ++i)
            {
                double temp = int(x[i] + 0.5);
                sum += temp * temp;
            }
            return sum;
        }
        string name()const{return "StepFunction" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(-100, 100));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct Weierstrass: public BaseF
    {//Nowhere differentiable quadratic; from Simon; separable (!? maybe typo in paper)
        int n;
        Weierstrass(int theN = 2): n(theN) {assert(theN % 2 == 0);}
        double operator()(Vector<double> const& x)const
        {
            int kMax = 20;
            double a = 0.5, b = 3, sum = 0, sum2 = 0;
            for(int k = 0; k <= kMax; ++k)
                sum2 += pow(a, k) * cos(PI() * pow(b, k));
            for(int i = 0; i < n; ++i)
            {
                double sum3 = 0;
                for(int k = 0; k <= kMax; ++k) sum3 += pow(a, k) *
                    cos(2 * PI() * pow(b, k) * (x[i] + 0.5));
                sum += sum3;
            }
            return sum - n * sum2;
        }
        string name()const{return "Weierstrass" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(-5, 5));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct Bukin6: public BaseF
    {//Interesting structure; from Jamil & Yang; non-separable
        double operator()(Vector<double> const& x)const
        {
            assert(x.getSize() == 2);
            return 100 * sqrt(abs(x[1] - 0.01 * x[0] * x[0])) +
                0.01 * abs(x[0] + 10);
        }
        string name()const{return "Bukin6";}
        Vector<pair<double, double> > getBox()const
        {
            Vector<pair<double, double> > box(2);
            box[0] = make_pair(-15, -5);
            box[1] = make_pair(-3, 3);
            return box;
        }
        Vector<double> getAnswer()const
        {
            Vector<double> answer(2);
            answer[0] = -10;
            answer[1] = 1;
            return answer;
        }
    };
    struct Damavandi: public BaseF
    {//DeceptiveQuadratic; from Jamil & Yang; non-separable
        double operator()(Vector<double> const& x)const
        {
            double temp = 1, sum = 0;
            for(int i = 0; i < 2; ++i)
            {
                sum += 2 + (x[i] - 7) * (x[i] - 7);
                double temp2 = PI() * (x[i] - 2);
                temp *= sin(temp2)/temp2;
            }
            return (1 + temp * temp * temp * temp * temp) * sum;
        }
        string name()const{return "Damavandi";}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(2, make_pair(0, 14));}
        Vector<double> getAnswer()const
            {return Vector<double>(2, 2);}
    };
    struct Easom: public BaseF
    {//Interesting structure; from Jamil & Yang; non-separable
        double operator()(Vector<double> const& x)const
        {
            assert(x.getSize() == 2);
            double temp1 = x[0] - PI(), temp2 = x[1] - PI();
            return -cos(x[0]) * cos(x[1]) * exp(-temp1 * temp1 -temp2 * temp2);
        }
        string name()const{return "Easom";}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(2, make_pair(-100, 100));}
        Vector<double> getAnswer()const
            {return Vector<double>(2, PI());}
    };
    struct GulfRND: public BaseF
    {//From More et al; paper has mistake "mi" should be "-" not m * (i + 1)
    //Real-world function; box from Jamil & Yang; non-separable
        double operator()(Vector<double> const& x)const
        {
            assert(x.getSize() == 3);
            Vector<double> fx(3);
            for(int i = 0; i < fx.getSize(); ++i)
            {
                double ti = (i + 1.0)/100, yi = 25 + pow(-50 * log(ti), 2.0/3);
                fx[i] = exp(-pow(abs(yi - x[1]), x[2])/x[0]) - ti;
            }
            return dotProduct(fx, fx);
        }
        string name()const{return "GulfRND";}
        Vector<pair<double, double> > getBox()const
        {
            Vector<pair<double, double> > box(3);
            box[0] = make_pair(0.1, 100);
            box[1] = make_pair(0, 25.6);
            box[2] = make_pair(0, 5);
            return box;
        }
        Vector<double> getX0()const
        {
            Vector<double> x0(3);
            x0[0] = 5;
            x0[1] = 2.5;
            x0[2] = 0.15;
            return x0;
        }
        Vector<double> getAnswer()const{return getX0() * 10;}
    };
    struct Price2: public BaseF
    {//Almost flat global landscape + noisy; from Jamil & Yang; non-separable
        double operator()(Vector<double> const& x)const
        {
            assert(x.getSize() == 2);
            double temp1 = sin(x[0]), temp2 = sin(x[1]);
            return 1 + temp1 * temp1 + temp2 * temp2 - 0.1 * exp(-dotProduct(x, x));
        }
        string name()const{return "Price2";}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(2, make_pair(-10, 10));}
        Vector<double> getAnswer()const{return Vector<double>(2, 0);}
    };
    struct Trefethen: public BaseF
    {//Interesting structure; from Jamil & Yang; non-separable
     //High-prec answer from Bornemann et al
        double operator()(Vector<double> const& x)const
        {
            assert(x.getSize() == 2);
            return exp(sin(50 * x[0])) + sin(60 * exp(x[1])) +
                sin(70 * sin(x[0])) + sin(sin(80 * x[1])) -
                sin(10 * (x[0] + x[1])) + dotProduct(x, x)/4;
        }
        string name()const{return "Trefethen";}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(2, make_pair(-10, 10));}
        Vector<double> getAnswer()const
        {
            Vector<double> answer(2);
            answer[0] = -0.024403079743737212;
            answer[1] = 0.21061242727591697;
            return answer;
        }
    };
    struct Trig: public BaseF
    {//From Dennis & Schnabel, they give no solution, but x = 0 works
    //Box from Jamil & Yang; non-separable
        int n;
        Trig(int theN = 2): n(theN) {}
        double operator()(Vector<double> const& x)const
        {
            double cSum = 0;
            for(int i = 0; i < n; ++i) cSum += cos(x[i]);
            Vector<double> fx(n, n - cSum);
            for(int i = 0; i < n; ++i)
                fx[i] += (i + 1) * (1 - cos(x[i])) - sin(x[i]);
            return dotProduct(fx, fx);
        }
        string name()const{return "Trig" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(0, PI()));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct Pinter: public BaseF
    {//Quadratic with much noise; From Jamil & Yang; non-separable
        int n;
        Pinter(int theN = 2): n(theN) {}
        double operator()(Vector<double> const& x)const
        {
            double sum = 0;
            for(int i = 0; i < n; ++i)
            {
                sum += (i + 1) + x[i] * x[i];
                double xm1 = x[(i - 1 + n) % n], xp1 = x[(i + 1) % n],
                    A = xm1 * sin(x[i]) + sin(xp1), temp = sin(A),
                    B = xm1 * xm1 - 2 * x[i] + 3 * xp1 - cos(x[i]) + 1;
                sum += 20 * (i + 1) * temp * temp;
                sum += (i + 1) * log10(1 + (i + 1) * B * B);
            }
            return sum;
        }
        string name()const{return "Pinter" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(-10, 10));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct Salomon: public BaseF
    {//NoisyQuadratic; From Jamil & Yang; non-separable
        int n;
        Salomon(int theN = 2): n(theN) {}
        double operator()(Vector<double> const& x)const
        {
            double temp = norm(x);
            return 1 - cos(2 * PI() * temp) + 0.1 * temp;
        }
        string name()const{return "Salomon" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(-100, 100));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct SchaeferF6: public BaseF
    {//Oscilatory; From Jamil & Yang; non-separable
        int n;
        SchaeferF6(int theN = 2): n(theN) {}
        double operator()(Vector<double> const& x)const
        {
            double sum = 0;
            for(int i = 0; i < n - 1; ++i)
            {
                double temp = x[i] * x[i] + x[i + 1] * x[i + 1],
                    temp1 = sin(sqrt(temp)),
                    temp2 = 1 + 0.001 * temp;
                sum += 0.5 + (temp1 * temp1 - 0.5)/(temp2 * temp2);
            }
            return sum;
        }
        string name()const{return "SchaeferF6" + toString(n);}
        Vector<pair<double, double> > getBox()const
            {return Vector<pair<double, double> >(n, make_pair(-100, 100));}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    static int evalCount;
    struct MetaF
    {
        shared_ptr<BaseF> f;
        template<typename F> MetaF(shared_ptr<F> const& theF): f(theF){}
        double operator()(Vector<double> const& x)const
        {
            ++evalCount;
            return (*f)(x);
        }
        string getName()const{return f->name();}
        Vector<pair<double, double> > getBox()const{return f->getBox();}
        pair<Vector<double>, double> getAnswer()const
            {return make_pair(f->getAnswer(), (*f)(f->getAnswer()));}
    };
    static Vector<MetaF> getFunctions()
    {
        Vector<MetaF> result;
        //result.append(MetaF(make_shared<Trig>(2)));
        result.append(MetaF(make_shared<Bukin6>()));
        result.append(MetaF(make_shared<Damavandi>()));
        result.append(MetaF(make_shared<Easom>()));
        result.append(MetaF(make_shared<GulfRND>()));
        result.append(MetaF(make_shared<Price2>()));
        result.append(MetaF(make_shared<Trefethen>()));
        int ds[] = {2, 6, 10, 30, 100};
        //int ds[] = {2, 6, 10, 30};
        for(int i = 0; i < sizeof(ds)/sizeof(ds[0]); ++i)
        {
            result.append(MetaF(make_shared<Ackley>(ds[i])));
            //result.append(MetaF(make_shared<FletcherPowell>(ds[i])));//very expensive omit for now
            result.append(MetaF(make_shared<Griewank>(ds[i])));
            result.append(MetaF(make_shared<Rastrigin>(ds[i])));
            result.append(MetaF(make_shared<SchwefelDoubleSum>(ds[i])));
            result.append(MetaF(make_shared<SchwefelMax>(ds[i])));
            result.append(MetaF(make_shared<StepFunction>(ds[i])));
            //result.append(MetaF(make_shared<Weierstrass>(ds[i])));//very expensive omit for now
            result.append(MetaF(make_shared<Trig>(ds[i])));
            result.append(MetaF(make_shared<Pinter>(ds[i])));
            result.append(MetaF(make_shared<Salomon>(ds[i])));
            result.append(MetaF(make_shared<SchaeferF6>(ds[i])));
        }
        return result;
    }
};
int TestFunctionsGlobalBoxMin::evalCount = 0;

template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    randomMinimize(FUNCTION const& f, SAMPLER const& s, int maxEvals = 1000000)
{
    assert(maxEvals > 0);
    pair<Vector<double>, double> best;
    for(int i = 0; i < maxEvals; ++i)
    {
        Vector<double> xNext = s();
        double yNext = f(xNext);
        if(i == 0 || !isfinite(best.second) || yNext < best.second)
        {
            best.first = xNext;
            best.second = yNext;
        }
    }
    return best;
}
template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    randomBeforeLocalMinimize(FUNCTION const& f, SAMPLER const& s,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(randomMinimize(f, s, maxEvals * 0.9).first,
        f, maxEvals * 0.1, yPrecision);
}

class ILSHybrid
{
    typedef pair<Vector<double>, double> P;
    template<typename FUNCTION, typename M_SAMPLER> struct Move
    {
        P current, best;
        FUNCTION const& f;
        M_SAMPLER s;
        int maxLocalEvals;
        void localSearchBest()
        {
            current = hybridLocalMinimize(current.first, f, maxLocalEvals,
                defaultPrecEps);
        }
        void bigMove()
        {//base jump on scale of best point
            Vector<double> xOld = current.first;
            current.first = s(current.first, best.first);//don't want escape
            if(!isfinite(norm(current.first))) current.first = xOld;
        }
        void updateBest()
        {
            if(!isfinite(best.second) || current.second < best.second)
                best = current;
        }
    };
public:
    template<typename FUNCTION, typename M_SAMPLER> static P minimize(
        Vector<double> const& initialGuess, M_SAMPLER const& s,
        FUNCTION const& f, int maxEvals = 1000000)
    {
        P initial(initialGuess, f(initialGuess));
        --maxEvals;
        int maxLocalEvals = sqrt(maxEvals);
        Move<FUNCTION, M_SAMPLER> move = {initial, initial, f, s,
            maxLocalEvals};
        iteratedLocalSearch(move, maxEvals/maxLocalEvals);
        return move.best;
    }
};

template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    differentialEvolutionReplacementMinimize(FUNCTION const& f, SAMPLER const& s,
    int maxEvals = 1000000)
{
    assert(maxEvals > 0);
    int n = 1;
    Vector<pair<Vector<double>, double> > population(n);
    for(int i = 0; i < n; ++i)
    {
        population[i].first = s();
        population[i].second = f(population[i].first);
    }
    maxEvals -= n;
    while(maxEvals > 0)
    {
        for(int i = 0; i < n && maxEvals-- > 0; ++i)
        {//mutate new point
            Vector<double> xiNew = population[i].first;
            int D = xiNew.getSize(), randK = GlobalRNG().mod(D);
            for(int k = 0; k < D; ++k) if(GlobalRNG().mod(2) || k == randK)
                xiNew[k] = s(k);
            //select best of original and mutated
            double yiNew = f(xiNew);
            if(yiNew < population[i].second)
            {
                population[i].first = xiNew;
                population[i].second = yiNew;
            }
        }
    }
    pair<Vector<double>, double>& best = population[0];
    for(int i = 1; i < n; ++i)
        if(best.second < population[i].second) best = population[i];
    return best;
}
template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    differentialEvolutionReplacementBeforeLocalMinimize(FUNCTION const& f, SAMPLER const& s,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(differentialEvolutionReplacementMinimize(f, s,
        maxEvals * 0.9).first, f, maxEvals * 0.1, yPrecision);
}

template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    differentialEvolutionBeforeLocalMinimize(FUNCTION const& f, SAMPLER const& s,
    Vector<pair<double, double> > const& box,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(differentialEvolutionMinimize(f, s, box,
        maxEvals * 0.9).first, f, maxEvals * 0.1, yPrecision);
}

template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    geneticAlgorithmMinimize(FUNCTION const& f, SAMPLER const& s,
    int maxEvals = 1000000, double mutationRate = 0.01)
{
    assert(maxEvals > 16);//need n >= 4
    int n = 2 * int(pow(maxEvals, 1.0/3)/2), nElite = 2;//need both even
    //int n = 2 * int(sqrt(maxEvals)/2), nElite = 2;//need both even
    typedef pair<double, Vector<double> > V;
    Vector<V> population(n), populationNew(n);
    for(int i = 0; i < n; ++i)
        population[i].first = f(population[i].second = s());
    maxEvals -= n;
    int D = population[0].second.getSize();
    //initialize rank roulette selector
    Vector<double> probabilities(n);
    for(int i = 0; i < n; ++i) probabilities[i] = n - i;
    probabilities *= 1.0/(n * (n + 1)/2);
    AliasMethod alias(probabilities);
    PairFirstComparator<double, Vector<double> > c;
    while((maxEvals -= n - nElite) >= 0)
    {
        quickSort(population.getArray(), 0, n - 1, c);
        //elitism - keep nElite best
        for(int j = 0; j < nElite; ++j) populationNew[j] = population[j];
        for(int i = nElite; i + 1 < n; i += 2)
        {//initialize children to picked parents
            for(int j = 0; j < 2; ++j)
                populationNew[i + j] = population[alias.next()];
            //uniform crossover and mutate
            for(int k = 0; k < D; ++k)
            {
                if(GlobalRNG().mod(2)) swap(populationNew[i].second[k],
                    populationNew[i + 1].second[k]);
                for(int j = 0; j < 2; ++j)
                    if(GlobalRNG().bernoulli(mutationRate))
                        populationNew[i + j].second[k] = s(k);
            }//evaluate
            for(int j = 0; j < 2; ++j) populationNew[i + j].first =
                f(populationNew[i + j].second);
        }
        population = populationNew;
    }
    int best = argMin(population.getArray(), n, c);
    return make_pair(population[best].second, population[best].first);
}
template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    geneticAlgorithmBeforeLocalMinimize(FUNCTION const& f, SAMPLER const& s,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(geneticAlgorithmMinimize(f, s,
        maxEvals * 0.9).first, f, maxEvals * 0.1, yPrecision);
}

//can define more general constraint then box later if needed
template<typename FUNCTION> pair<Vector<double>, double>
    Cholesky11CMA_ESMinimize(FUNCTION const& f, Vector<double> initialGuess,
    Vector<pair<double, double> > const& box, int maxEvals = 1000000)
{
    assert(maxEvals > 0);
    pair<Vector<double>, double> xy(initialGuess, f(initialGuess));
    --maxEvals;
    int D = initialGuess.getSize();
    Matrix<double> A = Matrix<double>::identity(D);
    double scale = max(1.0, norm(xy.first))/10, pst = 2.0/11, pt = 11.0/25,
        ca = sqrt(1 - 2.0/(D * D + 6)), cp = 1.0/12, d = 1 + 1.0/D,
        psAve = pst;
    while(maxEvals-- > 0)
    {
        Vector<double> z(D), xNext = xy.first;
        for(int i = 0; i < D; ++i) z[i] += GlobalRNG().normal01();
        xNext += A * z * scale;
        xNext = boxTrim(xNext, box);
        double yNext = f(xNext), ls = 0;
        if(yNext <= xy.second) ls = 1;
        psAve = (1 - cp) * psAve + cp * ls;
        scale *= exp((psAve - pst * (1 - psAve)/(1 - pst))/d);
        if(yNext <= xy.second)
        {
            xy.first = xNext;
            xy.second = yNext;
            if(psAve <= pt)
            {
                double z2 = dotProduct(z, z), ca2 = ca * ca;
                A = ca * (A + (sqrt(1 + (1 - ca2) * z2/ca2) - 1)/z2 *
                    outerProduct(A * z, z));
            }
        }
    }
    return xy;
}
template<typename FUNCTION> pair<Vector<double>,
    double> Cholesky11CMA_ESBeforeLocalMinimize(FUNCTION const& f,
    Vector<double> initialGuess, Vector<pair<double, double> > const& box,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(Cholesky11CMA_ESMinimize(f, initialGuess, box,
        maxEvals * 0.9).first, f, maxEvals * 0.1, yPrecision);
}

template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    RBCDBeforeLocalMinimize(FUNCTION const& f,
    SAMPLER const& s, int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(RBCDGeneral(f, s, s(), maxEvals * 0.9).first, f,
        maxEvals * 0.1, yPrecision);
}
template<typename FUNCTION, typename M_SAMPLER> pair<Vector<double>,
    double> simulatedAnnealingBeforeLocalMinimize(FUNCTION const& f,
    Vector<double> initialGuess, M_SAMPLER const& s,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(simulatedAnnealingMinimize(f, initialGuess, s,
        maxEvals * 0.9).first, f, maxEvals * 0.1, yPrecision);
}

template<typename FUNCTION, typename M_SAMPLER> pair<Vector<double>, double>
    simulatedAnnealingMinimize2(FUNCTION const& f, Vector<double> const& x0,
    M_SAMPLER const& s, int maxEvals = 1000000)
{
    pair<Vector<double>, double> xy(x0, f(x0)), xyBest = xy;
    --maxEvals;
    //starting solution x must be finite
    assert(isfinite(norm(xy.first)));
    double T = 0.1, coolingFactor = (maxEvals * 0.1 - 1)/(maxEvals * 0.1);
    for(; maxEvals--; T *= coolingFactor)
    {//steps and change are relative to best
        Vector<double> xNext = s(xy.first, xyBest.first);
        assert(isfinite(norm(xNext)));
        double yNext = f(xNext), relativeChange =
            (yNext - xy.second)/max(1.0, abs(xyBest.second));
        if(isnan(xy.second) || relativeChange < T * GlobalRNG().exponential(1))
        {
            xy.first = xNext;
            xy.second = yNext;
            if(isnan(xyBest.second) || yNext < xyBest.second) xyBest = xy;
        }
    }
    return xyBest;
}
template<typename FUNCTION, typename M_SAMPLER> pair<Vector<double>,
    double> simulatedAnnealingBeforeLocalMinimize2(FUNCTION const& f,
    Vector<double> initialGuess, M_SAMPLER const& s,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(simulatedAnnealingMinimize2(f, initialGuess, s,
        maxEvals * 0.9).first, f, maxEvals * 0.1, yPrecision);
}

template<typename FUNCTION, typename M_SAMPLER> pair<Vector<double>,
    double> basinHoppingMinimize(FUNCTION const& f, M_SAMPLER const& s,
    Vector<double> initialGuess, int maxEvals = 1000000)
{
    assert(maxEvals > 0);
    int localEvals = sqrt(maxEvals);
    pair<Vector<double>, double> xy = hybridLocalMinimize(initialGuess, f,
        localEvals);
    maxEvals -= localEvals;
    while(maxEvals-- > 0)
    {
        Vector<double> xNext = s(xy.first, xy.first);
        double yNext = f(xNext);
        if(yNext < xy.second)
        {
            xy = hybridLocalMinimize(xNext, f, localEvals);
            maxEvals -= localEvals;
        }

    }
    return xy;
}
template<typename FUNCTION, typename M_SAMPLER> pair<Vector<double>,
    double> basinHoppingBeforeLocalMinimize(FUNCTION const& f,
    M_SAMPLER const& s, Vector<double> initialGuess,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(basinHoppingMinimize(f, s, initialGuess,
        maxEvals * 0.9).first, f, maxEvals * 0.1, yPrecision);
}

template<typename FUNCTION, typename M_SAMPLER> pair<Vector<double>,
    double> markovianBeforeLocalMinimize(FUNCTION const& f,
    M_SAMPLER const& s, Vector<double> initialGuess,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(markovianMinimize(f, s, initialGuess,
        maxEvals * 0.9).first, f, maxEvals * 0.1, yPrecision);
}

template<typename FUNCTION, typename SAMPLER, typename M_SAMPLER>
pair<Vector<double>, double> mortalKombatMinimize(FUNCTION const& f,
    SAMPLER const& s, M_SAMPLER const& ms, int maxEvals = 1000000)
{
    assert(maxEvals > 0);
    int n = int(sqrt(maxEvals)/2) * 2;//must be even
    pair<Vector<double>, double> xy[2];
    for(int i = 0; i < 2; ++i)
    {
        xy[i].first = s();
        xy[i].second = f(xy[i].first);
        --maxEvals;
    }
    while((maxEvals -= 2) > 0)
    {
        for(int i = 0; i < 2; ++i)
        {
            Vector<double> xNext = ms(xy[i].first, xy[i].first);
            double yNext = f(xNext);
            if(yNext < xy[i].second)
            {
                xy[i].first = xNext;
                xy[i].second = yNext;
            }
        }
        if(maxEvals % n == 0)
        {
            if(xy[1].second < xy[0].second) swap(xy[0], xy[1]);
            xy[1].first = s();
            xy[1].second = f(xy[1].first);
            --maxEvals;
        }
    }
    if(xy[1].second < xy[0].second) swap(xy[0], xy[1]);
    return xy[0];
}
template<typename FUNCTION, typename SAMPLER, typename M_SAMPLER>
pair<Vector<double>, double> mortalKombatBeforeLocalMinimize(FUNCTION const& f,
    SAMPLER const& s, M_SAMPLER const& ms,
    int maxEvals = 1000000)
{
    assert(maxEvals > 1);
    return hybridLocalMinimize(mortalKombatMinimize(f, s, ms,
        maxEvals * 0.9).first, f, maxEvals * 0.1);
}

template<typename FUNCTION, typename SAMPLER, typename M_SAMPLER>
pair<Vector<double>, double> hybridMRBCDBeforeLocalMinimize(FUNCTION const& f,
    SAMPLER const& s, M_SAMPLER const& ms,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    assert(maxEvals > 100);//no point to have fewer
    return hybridLocalMinimize(RBCDGeneral(f, s,
        markovianMinimize(f, ms, s(),
        maxEvals * 0.45).first, maxEvals * 0.45).first, f, maxEvals * 0.1,
        yPrecision);
}

template<typename INCREMENTAL_FUNCTION, typename SAMPLER> double
    randomCoordinateDescent(INCREMENTAL_FUNCTION &f, SAMPLER const& s,
    int maxEvals = 1000000)
{
    int D = f.getSize();
    f.setCurrentDimension(0);
    double yLast = f(f.getXi());
    --maxEvals;
    while(f.getEvalCount() < maxEvals)
    {
        int j = GlobalRNG().mod(D);
        f.setCurrentDimension(j);
        double xjNew = s(j), yNew = f(xjNew);
        if(yNew < yLast)
        {
            yLast = yNew;
            f.bind(xjNew);
        }
    }
    return yLast;
}
template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    RCDGeneral(FUNCTION const &f, SAMPLER const& s, int maxEvals = 1000000)
{
    IncrementalWrapper<FUNCTION> iw(f, s());
    double y = randomCoordinateDescent(iw, s, maxEvals);
    return make_pair(iw.xBound, y);
}
template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    RCDBeforeLocalMinimize(FUNCTION const& f, SAMPLER const& s,
    int maxEvals = 1000000, double yPrecision = highPrecEps)
{
    return hybridLocalMinimize(RCDGeneral(f, s,
        maxEvals * 0.9).first, f, maxEvals * 0.1, yPrecision);
}

template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    incrementalRBCDBeforeLocalMinimizeGeneral(FUNCTION const &f,
    SAMPLER const& s, int maxEvals = 1000000)
{
    IncrementalWrapper<FUNCTION> iw(f, s());
    double y = incrementalRBCDBeforeLocalMinimize(iw, s, maxEvals);
    return make_pair(iw.xBound, y);
}

template<typename POINT, typename FUNCTION> void debugResultGlobalBox(
    pair<POINT, double> const& result,
    FUNCTION const& f, Vector<Vector<string> > & matrix, int start)
{
    debugResultHelper<TestFunctionsGlobalBoxMin>(result, f, matrix, start);
}
template<typename TESTCASE> void testAllSolversGlobalBox(TESTCASE const& f,
    Vector<Vector<string> >& matrix)
{
    int D = f.getAnswer().first.getSize(), start = 0;
    BoxSampler sb(f.getBox());
    Vector<double> initialGuess = sb();

    //box-aware data
    /*Vector<pair<double, double> > box = f.getBox();
    BoxSampler s(f.getBox());
    BoxConstrainedStepSampler ms(f.getBox());*/

    //agnostic data
    Vector<pair<double, double> > box = makeAgnosticBox(D);
    UnboundedSampler s(initialGuess);//to remove unfair advantage of 0
    AgnosticStepSampler ms;

    //benchmarks
    /*DEBUG("RandomBeforeLocalMinimize");
    matrix.lastItem().append("RandomBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(randomBeforeLocalMinimize(f, s), f, matrix, start);*/

    //Sobol only makes sense with a box!
    /*DEBUG("SobolBeforeLocalMinimize");
    matrix.lastItem().append("SobolBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(randomBeforeLocalMinimize(f, ScrambledSobolHybrid(f.getBox())), f, matrix, start);*/

    /*DEBUG("RCDBeforeLocalMinimize");
    matrix.lastItem().append("RCDBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(RCDBeforeLocalMinimize(f, s), f, matrix, start);
    //better samplers
    DEBUG("IncrementalRBCDBeforeLocalMinimizeGeneral");
    matrix.lastItem().append("IncrementalRBCDBeforeLocalMinimizeGeneral");
    start = clock();
    debugResultGlobalBox(incrementalRBCDBeforeLocalMinimizeGeneral(f, s), f, matrix, start);
    DEBUG("RBCDBeforeLocalMinimize");
    matrix.lastItem().append("RBCDBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(RBCDBeforeLocalMinimize(f, s), f, matrix, start);*/
    DEBUG("MarkovianBeforeLocalMinimize");
    matrix.lastItem().append("MarkovianBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(markovianBeforeLocalMinimize(f, ms, initialGuess), f, matrix, start);
    /*DEBUG("BasinHoppingBeforeLocalMinimize");
    matrix.lastItem().append("BasinHoppingBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(basinHoppingBeforeLocalMinimize(f, ms, initialGuess), f, matrix, start);
    DEBUG("MortalKombatBeforeLocalMinimize");
    matrix.lastItem().append("MortalKombatBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(mortalKombatBeforeLocalMinimize(f, s, ms), f, matrix, start);
    DEBUG("HybridBeforeLocalMinimize");
    matrix.lastItem().append("HybridBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(hybridBeforeLocalMinimize(f, s, ms), f, matrix, start);
    DEBUG("HybridMRBCDBeforeLocalMinimize");
    matrix.lastItem().append("HybridMRBCDBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(hybridMRBCDBeforeLocalMinimize(f, s, ms), f, matrix, start);
    DEBUG("HybridExplorativeBeforeLocalMinimize");
    matrix.lastItem().append("HybridExplorativeBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(hybridExplorativeBeforeLocalMinimize(f, s, ms, box), f, matrix, start);
    //metaheuristics
    DEBUG("ILSMinimize");
    matrix.lastItem().append("ILSMinimize");
    start = clock();
    debugResultGlobalBox(ILSHybrid::minimize(initialGuess, ms, f), f, matrix, start);
    //only test with box for now
    DEBUG("Cholesky11CMA_ESBeforeLocalMinimize");
    matrix.lastItem().append("Cholesky11CMA_ESBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(Cholesky11CMA_ESBeforeLocalMinimize(f, initialGuess, f.getBox()), f, matrix, start);*/
    DEBUG("SimulatedAnnealingBeforeLocalMinimize");
    matrix.lastItem().append("SimulatedAnnealingBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(simulatedAnnealingBeforeLocalMinimize(f, initialGuess, ms), f, matrix, start);
    DEBUG("SimulatedAnnealingBeforeLocalMinimize2");
    matrix.lastItem().append("SimulatedAnnealingBeforeLocalMinimize2");
    start = clock();
    debugResultGlobalBox(simulatedAnnealingBeforeLocalMinimize2(f, initialGuess, ms), f, matrix, start);
    /*DEBUG("DifferentialEvolutionBeforeLocalMinimize");
    matrix.lastItem().append("DifferentialEvolutionBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(differentialEvolutionBeforeLocalMinimize(f, s, box), f, matrix, start);
    DEBUG("DifferentialEvolutionReplacementBeforeLocalMinimize");
    matrix.lastItem().append("DifferentialEvolutionReplacementBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(differentialEvolutionReplacementBeforeLocalMinimize(f, s), f, matrix, start);
    DEBUG("GeneticAlgorithmBeforeLocalMinimize");
    matrix.lastItem().append("GeneticAlgorithmBeforeLocalMinimize");
    start = clock();
    debugResultGlobalBox(geneticAlgorithmBeforeLocalMinimize(f, s), f, matrix, start);*/
}

void testAllFunctionsGlobalBox()
{//ignore starting points given in test set
    Vector<Vector<string> > matrix;
    string name;
    Vector<TestFunctionsGlobalBoxMin::MetaF> fs =
        TestFunctionsGlobalBoxMin::getFunctions();
    int nRepeats = 1;
    //int nRepeats = 10;
    for(int i = 0; i < fs.getSize(); ++i) if(fs[i].getBox().getSize() <= 10)
    {
        for(int j = 0; j < nRepeats; ++j)
        {
            string name = fs[i].getName() + "_" + toString(j);
            DEBUG(name);
            matrix.append(Vector<string>());
            matrix.lastItem().append(name);
            testAllSolversGlobalBox(fs[i], matrix);
        }
    }
    createMinReport("reportMinGlobalBox", matrix, nRepeats);
}

template<typename TESTCASE> void testAllSolversGlobalBoxSmallBudget(TESTCASE const& f,
    Vector<Vector<string> >& matrix)
{
    int D = f.getAnswer().first.getSize(), start = 0, nEvals = 100;
    BoxSampler s(f.getBox());
    Vector<double> initialGuess = s();
    //UnboundedSampler s2(initialGuess, 1);//to remove unfair advantage of 0
    BoxConstrainedStepSampler ms(f.getBox());
    //AgnosticStepSampler<> ms;
    //benchmarks
    DEBUG("RandomMinimize");
    matrix.lastItem().append("RandomMinimize");
    start = clock();
    debugResultGlobalBox(randomMinimize(f, s, nEvals), f, matrix, start);
    DEBUG("SobolMinimize");
    matrix.lastItem().append("SobolMinimize");
    start = clock();
    debugResultGlobalBox(randomBeforeLocalMinimize(f, ScrambledSobolHybrid(f.getBox()), nEvals), f, matrix, start);
    //better samplers
    DEBUG("RBCDMinimize");
    matrix.lastItem().append("RBCDMinimize");
    start = clock();
    debugResultGlobalBox(RBCDGeneral(f, s, s(), nEvals), f, matrix, start);
    DEBUG("MarkovianMinimize");
    matrix.lastItem().append("MarkovianMinimize");
    start = clock();
    debugResultGlobalBox(markovianMinimize(f, ms, initialGuess, nEvals), f, matrix, start);
    DEBUG("SmallBudgetHybridMinimize");
    matrix.lastItem().append("SmallBudgetHybridMinimize");
    start = clock();
    debugResultGlobalBox(smallBudgetHybridMinimize(f, s, ms, nEvals), f, matrix, start);
}

void testAllFunctionsGlobalBoxSmallBudget()
{//ignore starting points given in test set
    Vector<Vector<string> > matrix;
    string name;
    Vector<TestFunctionsGlobalBoxMin::MetaF> fs =
        TestFunctionsGlobalBoxMin::getFunctions();
    int nRepeats = 10;
    for(int i = 0; i < fs.getSize(); ++i)
    {
        for(int j = 0; j < nRepeats; ++j)
        {
            string name = fs[i].getName() + "_" + toString(j);
            DEBUG(name);
            matrix.append(Vector<string>());
            matrix.lastItem().append(name);
            testAllSolversGlobalBoxSmallBudget(fs[i], matrix);
        }
    }
    createMinReport("reportMinGlobalBoxSmallBudget", matrix, nRepeats);
}

int main()
{
    testAllFunctionsGlobalBox();
    return 0;
    testAllFunctionsGlobalBoxSmallBudget();
    return 0;
    testAllFunctions();
    return 0;
    testAllFunctionsLargeD();
    return 0;
    testOpt1D();
    return 0;
}
