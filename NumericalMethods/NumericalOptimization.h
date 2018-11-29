#ifndef NUMERICAL_OPTIMIZATION_H
#define NUMERICAL_OPTIMIZATION_H
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
#include "NumericalMethods.h"
namespace igmdk{

template<typename FUNCTION> pair<double, double> minimizeGS(
    FUNCTION const& f, double xLeft, double xRight,
    double relAbsXPrecision = defaultPrecEps)
{//don't want precision too low
    assert(isfinite(xLeft) && isfinite(xRight) && xLeft <= xRight &&
        relAbsXPrecision >= numeric_limits<double>::epsilon());
    double GR = 0.618, xMiddle = xLeft * GR + xRight * (1 - GR),
        yMiddle = f(xMiddle);
    while(isELess(xLeft, xRight, relAbsXPrecision))
    {
        bool chooseR = xRight - xMiddle > xMiddle - xLeft;
        double xNew = GR * xMiddle + (1 - GR) *
            (chooseR ? xRight : xLeft), yNew = f(xNew);
        if(yNew < yMiddle)
        {
            (chooseR ? xLeft : xRight) = xMiddle;
            xMiddle = xNew;
            yMiddle = yNew;
        }
        else (chooseR ? xRight : xLeft) = xNew;
    }
    return make_pair(xMiddle, yMiddle);
}
int roundToNearestInt(double x)
{
    return int(x > 0 ? x + 0.5 : x - 0.5);
}
template<typename FUNCTION> pair<int, double> minimizeGSDiscrete(
    FUNCTION const& f, int xLeft, int xRight)
{
    assert(isfinite(xLeft) && isfinite(xRight) && xLeft <= xRight);
    double GR = 0.618;
    int xMiddle = roundToNearestInt(xLeft * GR + xRight * (1 - GR));
    double yMiddle = f(xMiddle);
    while(xLeft < xRight)
    {
        bool chooseR = xRight - xMiddle > xMiddle - xLeft;
        int xNew = roundToNearestInt(GR * xMiddle + (1 - GR) *
            (chooseR ? xRight : xLeft));
        double yNew = xNew == xMiddle ? yMiddle : f(xNew);
        if(yNew < yMiddle)
        {
            (chooseR ? xLeft : xRight) = xMiddle;
            xMiddle = xNew;
            yMiddle = yNew;
        }
        else (chooseR ? xRight : xLeft) = xNew;
    }
    return make_pair(xMiddle, yMiddle);
}

template<typename FUNCTION> pair<double, double> unimodalMinBracket(FUNCTION
    const& f, double x0, double fx, bool twoSided, double d, int maxEvals)
{
    assert(abs(d) > 0 && isfinite(x0) && isfinite(d));// && maxEvals > 2?
    pair<double, double> best(x0, x0 + d);
    double fMin = f(x0 + d);
    maxEvals -= 2;
    if(fx < fMin && twoSided)//check decrease direction if 2-sided
    {
        d *= -1;//maximal pattern must be in the other direction
        fMin = fx;
        swap(best.first, best.second);
    }
    if(!(fx < fMin))//if 1-sided can't increase current bracket
        while(d * 2 != d && maxEvals-- > 0)//handle d = 0, inf, and NaN
        {
            d *= 2;
            double xNext = x0 + d, fNext = f(xNext);
            if(fNext < fMin)
            {//shift
                best.first = best.second;
                best.second = xNext;
                fMin = fNext;
            }
            else
            {//found 3-point pattern, form interval
                best.second = xNext;
                break;
            }
        }//ensure sorted interval
    if(best.first > best.second) swap(best.first, best.second);
    return best;
}

template<typename FUNCTION> pair<double, double> minimizeGSBracket(
    FUNCTION const& f, double x, double fx = numeric_limits<double>::
    quiet_NaN(), bool twoSided = true, double step = 0.1,
    double relAbsXPrecision = defaultPrecEps, int bracketMaxEvals = 100)
{
    if(!isfinite(x)) return make_pair(x, fx);
    if(!isfinite(fx))
    {
        fx = f(x);
        --bracketMaxEvals;
    }
    pair<double, double> bracket = unimodalMinBracket(f, x, fx, twoSided,
        step * max(1.0, abs(x)), bracketMaxEvals), result = minimizeGS(f,
        bracket.first, bracket.second, relAbsXPrecision);//ensure nondecrease
    return result.second < fx ? result : make_pair(x, fx);
}

template<typename FUNCTION> struct IncrementalWrapper
{
    FUNCTION f;
    mutable Vector<double> xBound;
    int i;
    mutable int evalCount;
public:
    IncrementalWrapper(FUNCTION const& theF, Vector<double> const& x0):
        f(theF), xBound(x0), i(-1), evalCount(0) {}
    void setCurrentDimension(double theI)
    {
        assert(theI >= 0 && theI < xBound.getSize());
        i = theI;
    }
    int getEvalCount()const{return evalCount;}
    int getSize()const{return xBound.getSize();}
    double getXi()const{return xBound[i];}
    double operator()(double xi)const
    {
        double oldXi = xBound[i];
        xBound[i] = xi;
        double result = f(xBound);
        ++evalCount;
        xBound[i] = oldXi;
        return result;
    }
    void bind(double xi)const{xBound[i] = xi;}
};

template<typename INCREMENTAL_FUNCTION> double unimodalCoordinateDescent(
    INCREMENTAL_FUNCTION &f, int maxEvals = 1000000,
    double xPrecision = defaultPrecEps)
{
    int D = f.getSize();
    Vector<int> order(D);
    for(int i = 0; i < D; ++i) order[i] = i;
    Vector<double> steps(D, 0.1);
    double yLast = numeric_limits<double>::quiet_NaN();
    bool done = false;
    while(!done && f.getEvalCount() < maxEvals)//may be exceeded but ok
    {//use random order full cycles
        GlobalRNG().randomPermutation(order.getArray(), D);
        done = true;
        double yNow = yLast;
        for(int i = 0; i < D && f.getEvalCount() < maxEvals; ++i)
        {
            int j = order[i];
            f.setCurrentDimension(j);
            pair<double, double> resultJ = minimizeGSBracket(f, f.getXi(),
                yLast, true, steps[j], xPrecision);
            double dx = resultJ.first - f.getXi();
            if(abs(dx) > max(1.0, abs(f.getXi())) * xPrecision)
            {
                done = false;
                steps[j] = dx;
            }
            f.bind(resultJ.first);
            yLast = resultJ.second;
        }
        if(isfinite(yNow) && yLast >= yNow) done = true;
    }
    return yLast;
}
template<typename FUNCTION> pair<Vector<double>, double>
    unimodalCoordinateDescentGeneral(FUNCTION const& f, Vector<double> const&
    x0, int maxEvals = 1000000, double xPrecision = defaultPrecEps)
{
    IncrementalWrapper<FUNCTION> iw(f, x0);
    double y = unimodalCoordinateDescent(iw, maxEvals, xPrecision);
    return make_pair(iw.xBound, y);
}

double steepestStepScale(Vector<double> const& x, double y,
    Vector<double> const& grad)
{
    return max(1/max(1.0, abs(y)), max(1.0, norm(x)) *
        sqrt(numeric_limits<double>::epsilon())/norm(grad));
}

template<typename FUNCTION, typename DIRECTIONAL_DERIVATIVE> bool
    strongWolfeLineSearchMoreThuente(FUNCTION const& f,
    Vector<double> const& gradient, DIRECTIONAL_DERIVATIVE const& dd,
    Vector<double>& x, double& fx, int& maxEvals, Vector<double> const& dx,
    double yEps)
{
    double dd0 = dotProduct(dx, gradient), sLo = 0, fLo = fx, s = 1,
        sHi = numeric_limits<double>::infinity(), temp = 0.0001 * dd0;
    if(!isfinite(dd0) || dd0 >= 0) return true;
    while(maxEvals > 0 && isfinite(s) &&
        (!isfinite(sHi) || isELess(fx + dd0 * abs(sHi - sLo), fx, yEps)))
    {
        double fNew = f(x + dx * s);
        if(!isfinite(fNew)) break;
        --maxEvals;
        if(fNew - s * temp > fLo - sLo * temp) sHi = s;//case 1
        else
        {
            double ddS = dd(x + dx * s, dx);
            maxEvals -= dd.fEvals();
            if(abs(ddS) <= -0.1 * dd0)//check for early termination
            {
                sLo = s;
                fLo = fNew;
                break;
            }
            if((ddS - temp) * (sLo - s) <= 0) sHi = sLo;//case 3
            //case 2 and 3
            sLo = s;
            fLo = fNew;
        }
        if(isfinite(sHi)) s = (sLo + sHi)/2;//zooming
        else s *= 2;//case 2 init doubling
    }
    if(sLo > 0)
    {//any non-0 guaranteed to have sufficient descent
        x += dx * sLo;
        double fxFirst = fx;
        fx = fLo;
        return !isELess(fx, fxFirst, yEps);//must make good progress
    }
    return true;
}

template<typename FUNCTION1D> struct EvalCountWrapper
{//to keep track of used evaluations, which golden section doesn't
    FUNCTION1D f;
    mutable int evalCount;
    EvalCountWrapper(FUNCTION1D const& theF): f(theF), evalCount(0){}
    double operator()(double x)const
    {
        ++evalCount;
        return f(x);
    }
};
template<typename FUNCTION, typename DIRECTIONAL_DERIVATIVE> bool
    goldenSectionLineSearch(FUNCTION const& f, Vector<double> const& gradient,
    DIRECTIONAL_DERIVATIVE const& dd, Vector<double>& x, double& fx,
    int& maxEvals, Vector<double> const& dx, double yEps, bool useExact = true)
{
    if(!(norm(dx) > 0)) return false;//this way handle NaN also
    double step = 1;//well-scaled for most algorithms
    if(useExact)
    {
        EvalCountWrapper<ScaledDirectionFunction<FUNCTION> > f2(
            ScaledDirectionFunction<FUNCTION>(f, x, dx));
        pair<double, double> result = minimizeGSBracket(f2, f2.f.getS0(), fx,
            false);
        maxEvals -= f2.evalCount;//may be exceeded but ok
        if(isELess(result.second, fx, yEps))
            step = result.first - f2.f.getS0();
    }//ensure strong Wolfe conditions
    return strongWolfeLineSearchMoreThuente(f, gradient, dd, x, fx, maxEvals,
        dx * step, yEps);
}

template<typename FUNCTION, typename GRADIENT, typename DIRECTIONAL_DERIVATIVE>
    pair<Vector<double>, double> LBFGSMinimize(Vector<double> const& x0,
    FUNCTION const& f, GRADIENT const& g, DIRECTIONAL_DERIVATIVE const& dd,
    int maxEvals = 1000000, double yPrecision = highPrecEps,
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
            d = grad * (-steepestStepScale(xy.first, xy.second, grad));
            history = Queue<pair<V, V> >();
        }
        pair<V, double> xyOld = xy;
        if(goldenSectionLineSearch(f, grad, dd, xy.first, xy.second, maxEvals,
            d, yPrecision, useExact))
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
        d *= 1/(dotProduct(history[last].second, history[last].second) *
            p[last]);
        for(int i = 0; i < history.getSize(); ++i)
        {
            double bi = dotProduct(history[i].second, d) * p[last - i];
            d += history[i].first * (a[last - i] - bi);
        }
        d *= -1;
    }
    return xy;
}

template<typename FUNCTION> struct NelderMead
{
    FUNCTION f;
    int D;
    Vector<double> vertexSum;//incremental centroid
    typedef pair<Vector<double>, double> P;
    Vector<P> simplex;
    double scale(P& high, double factor, int& maxEvals)
    {
        P result = high;
        //affine combination of the high point and the
        //centroid of the remaining vertices
        //centroid = (vertexSum - high)/D and
        //result = centroid * (1 - factor) + high * factor
        double centroidFactor = (1 - factor)/D;
        result.first = vertexSum * centroidFactor +
            high.first * (factor - centroidFactor);
        result.second = f(result.first);
        --maxEvals;
        if(result.second < high.second)
        {//accept scaling if improving
            vertexSum += result.first - high.first;
            high = result;
        }
        return result.second;
    }
public:
    NelderMead(int theD, FUNCTION const& theFunction = FUNCTION()):
        D(theD), f(theFunction), simplex(D + 1){}

    P minimize(Vector<double> const& initialGuess, int maxEvals = 1000000,
        double yPrecision = highPrecEps, double step = 1)
    {//initialize the simplex
        vertexSum = initialGuess;
        for(int i = 0; i < D; ++i) vertexSum[i] = 0;
        for(int i = 0; i <= D; ++i)
        {
            simplex[i].first = initialGuess;
            if(i > 0) simplex[i].first[i - 1] += GlobalRNG().uniform01() *
                step * max(1.0, abs(initialGuess[i - 1]))/10;
            simplex[i].second = f(simplex[i].first);
            --maxEvals;
            vertexSum += simplex[i].first;
        }
        for(;;)
        {//calculate high, low, and nextHigh, which must be all different
            int high = 0, nextHigh = 1, low = 2;
            if(simplex[high].second < simplex[nextHigh].second)
                swap(high, nextHigh);
            if(simplex[nextHigh].second < simplex[low].second)
            {
                swap(low, nextHigh);
                if(simplex[high].second < simplex[nextHigh].second)
                    swap(high, nextHigh);
            }
            for(int i = 3; i <= D; ++i)
            {
                if(simplex[i].second < simplex[low].second) low = i;
                else if(simplex[i].second > simplex[high].second)
                {
                    nextHigh = high;
                    high = i;
                }
                else if(simplex[i].second > simplex[nextHigh].second)
                    nextHigh = i;
            }//check if already converged
            if(maxEvals <= 0 || !isELess(simplex[low].second,
                simplex[high].second, yPrecision)) return simplex[low];
            //try to reflect
            double value = scale(simplex[high], -1, maxEvals);
            //try to double if better than low
            if(value <= simplex[low].second) scale(simplex[high], 2, maxEvals);
            else if(value >= simplex[nextHigh].second)
            {//try reflected/unrefrected halving if accepted/rejected value
                double yHi = simplex[high].second;
                if(scale(simplex[high], 0.5, maxEvals) >= yHi)
                {//contract all to get rid of the high point
                    vertexSum = simplex[low].first;
                    for(int i = 0; i <= D; ++i) if(i != low)
                    {
                        vertexSum += simplex[i].first = (simplex[i].first +
                            simplex[low].first) * 0.5;
                        simplex[i].second = f(simplex[i].first);
                        --maxEvals;
                    }
                }
            }
        }
    }

    P restartedMinimize(Vector<double> const& initialGuess,
        int maxEvals = 100000, double yPrecision = highPrecEps,
        int maxRepeats = 10, double step = 1)
    {
        P result(initialGuess, numeric_limits<double>::infinity());
        while(maxRepeats--)
        {
            double yOld = result.second;
            result = minimize(result.first, maxEvals, yPrecision, step);
            if(!isELess(result.second, yOld, yPrecision)) break;
        }
        return result;
    }
};

template<typename FUNCTION> Vector<double> gridMinimize(
    Vector<Vector<double> > const& sets, FUNCTION const& f = FUNCTION())
{
    assert(sets.getSize() > 0);
    long long total = 1;
    for(int i = 0; i < sets.getSize(); ++i)
    {
        assert(sets[i].getSize() > 0);
        total *= sets[i].getSize();
    }
    Vector<double> best;
    double bestScore;
    for(long long i = 0; i < total; ++i)
    {//unrank and eval
        long long rank = i;
        Vector<double> next;
        for(int j = 0; j < sets.getSize(); ++j)
        {
            next.append(sets[j][rank % sets[j].getSize()]);
            rank /= sets[j].getSize();
        }
        double score = f(next);
        if(best.getSize() == 0 || score < bestScore)
        {
            bestScore = score;
            best = next;
        }
    }
    return best;
}

template<typename FUNCTION> Vector<double> randomDiscreteMinimize(
    Vector<Vector<double> > const& sets, FUNCTION const& f = FUNCTION(),
    int evals = 20)
{
    assert(sets.getSize() > 0);
    Vector<double> best;
    double bestScore;
    for(long long i = 0; i < evals; ++i)
    {
        Vector<double> next;
        for(int j = 0; j < sets.getSize(); ++j)
            next.append(sets[j][GlobalRNG().mod(sets[j].getSize())]);
        double score = f(next);
        if(best.getSize() == 0 || score < bestScore)
        {
            bestScore = score;
            best = next;
        }
    }
    return best;
}

template<typename FUNCTION> Vector<double> compassDiscreteMinimize(
    Vector<Vector<double> > const& sets, FUNCTION const& f = FUNCTION(),
    int remainingEvals = 100)
{//use median in each set as initial solution
    Vector<int> current;
    for(int i = 0; i < sets.getSize(); ++i)
    {
        assert(sets[i].getSize() > 0);
        current.append(sets[i].getSize()/2);
    }
    return compassDiscreteMinimizeHelper(sets, current, f,
        remainingEvals).first;
}
//assumes set values are in sorted (or reverse sorted) order!
template<typename FUNCTION> pair<Vector<double>, pair<double, int> >
    compassDiscreteMinimizeHelper(Vector<Vector<double> > const& sets,
    Vector<int> current, FUNCTION const& f = FUNCTION(),
    int remainingEvals = 100)
{//start with medians
    Vector<double> best;
    for(int i = 0; i < sets.getSize(); ++i)
    {
        assert(0 <= current[i] && current[i] < sets[i].getSize());
        best.append(sets[i][current[i]]);
    }
    double bestScore = f(best);
    Vector<int> preferredSign(sets.getSize(), 1);
    for(bool isOpt = false, changedSign = false; !isOpt;)
    {
        isOpt = true;
        for(int i = 0; i < sets.getSize(); ++i)
        {
            int next = current[i] + preferredSign[i];
            if(0 <= next && next < sets[i].getSize())
            {
                if(remainingEvals-- < 1)
                    return make_pair(best, make_pair(bestScore, 0));
                best[i] = sets[i][next];
                double score = f(best);
                if(score < bestScore)
                {
                    current[i] = next;
                    bestScore = score;
                    isOpt = false;
                }
                else
                {
                    best[i] = sets[i][current[i]];
                    preferredSign[i] *= -1;
                }
            }
            else preferredSign[i] *= -1;
        }
        if(isOpt){if(!changedSign) isOpt = false;}
        else changedSign = false;
    }
    return make_pair(best, make_pair(bestScore, remainingEvals));
}

double RMRate(int i){return 1/pow(i + 1, 0.501);}

template<typename POINT, typename FUNCTION> POINT SPSA(POINT x,
    FUNCTION const& f, int maxEvals = 10000, double initialStep = 1)
{
    POINT direction = x;
    for(int i = 0, D = x.getSize(); i < maxEvals/2; ++i)
    {
        for(int j = 0; j < D; ++j) direction[j] =
            GlobalRNG().next() % 2 ? 1 : -1;
        double step = initialStep/pow(i + 1, 0.101), temp = RMRate(i) *
            (f(x + direction * step) - f(x - direction * step))/2;
        if(!isfinite(temp)) break;
        for(int j = 0; j < D; ++j) x[j] -= temp/direction[j];
    }
    return x;
}
template<typename POINT, typename FUNCTION> pair<POINT, double> metaSPSA(
    POINT x, FUNCTION const& f, int spsaEvals = 100000, int estimateEvals =
    100, double step = pow(2, 10), double minStep = pow(2, -20))
{
    pair<POINT, double> xy(x, numeric_limits<double>::infinity());
    for(; step > minStep; step /= 2)
    {
        if(isfinite(xy.second)) x = SPSA(xy.first, f, spsaEvals, step);
        double sum = 0;
        for(int i = 0; i < estimateEvals; ++i) sum += f(x);
        if(sum/estimateEvals < xy.second)
        {
            xy.first = x;
            xy.second = sum/estimateEvals;
        }
    }
    return xy;
}

class ForcingFunction
{
    double a;
public:
    ForcingFunction(double scale): a(0.0001 * scale){assert(scale > 0);}
    double operator()(double s)const{return a * s * s;}
};
template<typename INCREMENTAL_FUNCTION> double compassMinimizeIncremental(
    INCREMENTAL_FUNCTION &f, int maxEvals = 1000000,
    double xPrecision = highPrecEps)
{//use scaled step
    f.setCurrentDimension(0);
    double yBest = f(f.getXi()), step = 0.1;
    int D = f.getSize(), nD = 2 * D, nCycleEvals = 0;
    Vector<int> order(nD);
    for(int i = 0; i < nD; ++i) order[i] = i;
    GlobalRNG().randomPermutation(order.getArray(), nD);
    for(; step > xPrecision && f.getEvalCount() < maxEvals; step /= 2)
    {//poll in all directions in random order
        for(int i = 0; i < nD && f.getEvalCount() < maxEvals; ++i)
        {
            int d = order[nCycleEvals++ % nD], j = d/2, sign = d % 2 ? 1 : -1;
            f.setCurrentDimension(j);
            double scale = max(1.0, abs(f.getXi())),
                xjNew = f.getXi() + sign * step * scale, yNew = f(xjNew);
            ForcingFunction ff(max(1.0, (isfinite(yBest) ? abs(yBest) : 0)));
            if((!isfinite(yBest) && isfinite(yNew)) || yBest - yNew > ff(step))
            {//found good enough step
                f.bind(xjNew);
                yBest = yNew;
                step *= 4;
                break;
            }
        }
        if(nCycleEvals >= nD)//permute only when had enough evals and not
        {//during a cycle
            GlobalRNG().randomPermutation(order.getArray(), nD);
            nCycleEvals = 0;
        }
    }
    return yBest;
}
template<typename FUNCTION> pair<Vector<double>, double> compassMinimize(
    Vector<double> const& x0, FUNCTION const& f, int maxEvals = 1000000,
    double xPrecision = highPrecEps)
{
    IncrementalWrapper<FUNCTION> iw(f, x0);
    double y = compassMinimizeIncremental(iw, maxEvals, xPrecision);
    return make_pair(iw.xBound, y);
}

template<typename FUNCTION> struct Vector1DFunction
{
    FUNCTION f;
    Vector1DFunction(FUNCTION const& theF): f(theF){}
    double operator()(Vector<double> const& x)const
    {
        assert(x.getSize() == 1);
        return f(x[0]);
    }
};

template<typename FUNCTION> pair<double, double> compassMinimize1D(
    double x0, FUNCTION const& f, int maxEvals = 1000000,
    double xPrecision = highPrecEps)
{
    Vector<double> x(1, x0);
    pair<Vector<double>, double> result = compassMinimize(x,
        Vector1DFunction<FUNCTION>(f), maxEvals, xPrecision);
    return make_pair(result.first[0], result.second);
}

template<typename FUNCTION, typename GRADIENT, typename DIRECTIONAL_DERIVATIVE>
    pair<Vector<double>, double> gradDescent(Vector<double> const& x0,
    FUNCTION const& f, GRADIENT const& g, DIRECTIONAL_DERIVATIVE const& dd,
    int maxEvals = 1000000, double yPrecision = highPrecEps,
    bool useExact = true)
{
    pair<Vector<double>, double> xy(x0, f(x0));
    while((maxEvals -= g.fEvals(xy.first.getSize()) + 1) > 0)
    {
        Vector<double> grad = g(xy.first);
        bool failed = goldenSectionLineSearch(f, grad, dd, xy.first, xy.second,
            maxEvals, grad * (-steepestStepScale(xy.first, xy.second, grad)),
            yPrecision, useExact);
        if(failed) break;
    }
    return xy;
}

/*
template<typename FUNCTION> struct Grad1DFunction
{
    FUNCTION f;
    Grad1DFunction(FUNCTION const& theF): f(theF){}
    Vector<double> operator()(Vector<double> const& x)const
    {
        assert(x.getSize() == 1);
        return Vector<double>(1, f(x[0]));
    }
    int fEvals(int D)const{return f.fEvals();}
};
template<typename FUNCTION, typename GRAD1D> pair<double, double>
    gradientDescentDBT1D(double x0, FUNCTION const& f,
    GRAD1D const& g, int maxEvals = 1000000,
    double xPrecision = defaultPrecEps)
{
    Vector<double> x(1, x0);
    pair<Vector<double>, double> result = gradDescent(x,
        Vector1DFunction<FUNCTION>(f), Grad1DFunction<GRAD1D>(g),
        maxEvals, xPrecision, false);
    return make_pair(result.first[0], result.second);
}*/

template<typename FUNCTION, typename SUBGRADIENT> pair<Vector<double>, double>
    subgradientDescent(Vector<double> const& x0, FUNCTION const& f,
    SUBGRADIENT const& g, int maxEvals = 1000000)
{
    pair<Vector<double>, double> xy(x0, f(x0));
    int stepCount = 1;
    while((maxEvals -= g.fEvals(xy.first.getSize()) + 1) > 0)
    {
        Vector<double> subgrad = g(xy.first), step = subgrad * (-max(1.0,
            norm(xy.first))/10/norm(subgrad)/stepCount),
            xNew = xy.first + step;
        double yNew = f(xNew);
        if(isfinite(yNew) && isfinite(normInf(xNew)))
        {
            xy.first = xNew;
            xy.second = yNew;
        }
    }
    return xy;
}

template<typename FUNCTION, typename SUBGRADIENT> pair<Vector<double>, double>
    subgradientDescent2(Vector<double> const& x0, FUNCTION const& f,
    SUBGRADIENT const& g, int maxEvals = 1000000)
{
    pair<Vector<double>, double> xy(x0, f(x0));
    int stepCount = 1;
    while((maxEvals -= g.fEvals(xy.first.getSize()) + 1) > 0)
    {
        Vector<double> subgrad = g(xy.first), step = subgrad *
            (-steepestStepScale(xy.first, xy.second, subgrad)/sqrt(stepCount)),
            xNew = xy.first + step;
        double yNew = f(xNew);
        if(isfinite(yNew) && isfinite(normInf(xNew)))
        {
            xy.first = xNew;
            xy.second = yNew;
        }
    }
    return xy;
}

template<typename FUNCTION> pair<Vector<double>, double> hybridLocalMinimize(
    Vector<double> const& x0, FUNCTION const& f, int maxEvals = 1000000,
    double yPrecision = highPrecEps)
{
    GradientFunctor<FUNCTION> g(f);
    DirectionalDerivativeFunctor<FUNCTION> dd(f);
    int LBFGSevals = x0.getSize() < 200 ? maxEvals/2 : maxEvals;
    pair<Vector<double>, double> result = LBFGSMinimize(x0, f, g, dd,
       LBFGSevals, yPrecision);
    if(x0.getSize() < 200)
    {
        int nRestarts = 30;
        NelderMead<FUNCTION> nm(x0.getSize(), f);
        result = nm.restartedMinimize(result.first,
            (maxEvals - LBFGSevals)/nRestarts, highPrecEps, nRestarts);
    }
    return result;
}

struct AgnosticStepSampler
{
    Vector<double> operator()(Vector<double> const& x,
        Vector<double> const& xScale)const
    {//ensure finite samples
        assert(isfinite(norm(x)) && isfinite(norm(xScale)));
        Vector<double> u = GlobalRNG().randomUnitVector(x.getSize()),
            result(x.getSize(), numeric_limits<double>::infinity());
        while(!isfinite(norm(result))) result =  x + u *
            (findDirectionScale(xScale, u)/10 * GlobalRNG().Levy());
        return result;
    }
};
class UnboundedSampler
{
    Vector<double> center;
    double scaleFactor;
public:
    UnboundedSampler(Vector<double> const& theCenter, double theScaleFactor =
        10): center(theCenter), scaleFactor(theScaleFactor)
        {assert(isfinite(norm(theCenter)));}
    Vector<double> operator()()const
    {
        Vector<double> next = center;
        for(int i = 0; i < next.getSize(); ++i) next[i] = (*this)(i);
        return next;
    }
    double operator()(int i)const
    {//ensure finite samples
        double result = numeric_limits<double>::infinity();
        while(!isfinite(result)) result = center[i] + max(1.0, abs(center[i]))/
            10 * scaleFactor * GlobalRNG().Levy() * GlobalRNG().sign();
        return result;
    }
};
class BoxSampler
{
    Vector<pair<double, double> > box;
public:
    BoxSampler(Vector<pair<double, double> > const& theBox): box(theBox){}
    Vector<double> operator()()const
    {
        Vector<double> next(box.getSize());
        for(int i = 0; i < box.getSize(); ++i) next[i] = (*this)(i);
        return next;
    }
    double operator()(int i)const
        {return GlobalRNG().uniform(box[i].first, box[i].second);}
};

template<typename INCREMENTAL_FUNCTION, typename SAMPLER> double
    randomBlockCoordinateDescent(INCREMENTAL_FUNCTION &f, SAMPLER const& s,
    int maxEvals = 1000000)
{
    int D = f.getSize();
    f.setCurrentDimension(0);
    double yLast = f(f.getXi());
    --maxEvals;
    while(f.getEvalCount() < maxEvals)
    {
        int k = min(D, GlobalRNG().geometric05());
        Vector<int> dims = GlobalRNG().randomCombination(k, D);
        Vector<double> oldXi(k);
        for(int i = 0; i < k; ++i)
        {
            int j = dims[i];
            f.setCurrentDimension(j);
            oldXi[i] = f.getXi();
            f.bind(s(j));
        }
        double yNew = f(f.getXi());
        if(!(isnan(yLast) || yNew < yLast))
            for(int i = 0; i < k; ++i)
            {
                f.setCurrentDimension(dims[i]);
                f.bind(oldXi[i]);
            }
        else yLast = yNew;
    }
    return yLast;
}
template<typename INCREMENTAL_FUNCTION, typename SAMPLER>
    double incrementalRBCDBeforeLocalMinimize(
    INCREMENTAL_FUNCTION &f, SAMPLER const& s, int maxEvals = 1000000,
    double xPrecision = highPrecEps)
{
    randomBlockCoordinateDescent(f, s, maxEvals * 0.9);
    //f remembers evals so far
    return compassMinimizeIncremental(f, maxEvals, xPrecision);
}

template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    RBCDGeneral(FUNCTION const &f, SAMPLER const& s,
    Vector<double> x0 = Vector<double>(), int maxEvals = 1000000)
{
    if(x0.getSize() == 0) x0 = s();
    IncrementalWrapper<FUNCTION> iw(f, x0);
    double y = randomBlockCoordinateDescent(iw, s, maxEvals);
    return make_pair(iw.xBound, y);
}

Vector<double> boxTrim(Vector<double> x,
    Vector<pair<double, double> > const& box)
{
    for(int i = 0; i < x.getSize(); ++i)
    {
        if(x[i] < box[i].first) x[i] = box[i].first;
        else if(isnan(x[i]) || x[i] > box[i].second) x[i] = box[i].second;
    }
    return x;
}
struct BoxConstrainedStepSampler
{
    Vector<pair<double, double> > box;
    AgnosticStepSampler s;
    BoxConstrainedStepSampler(Vector<pair<double, double> > const& theBox):
        box(theBox){}
    Vector<double> operator()(Vector<double> const& x,
        Vector<double> const& xScale)const{return boxTrim(s(x, xScale), box);}
};

template<typename FUNCTION, typename M_SAMPLER> pair<Vector<double>, double>
    simulatedAnnealingMinimize(FUNCTION const& f, Vector<double> const& x0,
    M_SAMPLER const& s, int maxEvals = 1000000)
{
    pair<Vector<double>, double> xy(x0, f(x0)), xyBest = xy;
    --maxEvals;
    //starting solution x must be finite
    assert(isfinite(norm(xy.first)));
    double T = 0.1, coolingFactor = (maxEvals * 0.2 - 1)/(maxEvals * 0.2);
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

Vector<pair<double, double> > makeAgnosticBox(int D)
{
    double inf = numeric_limits<double>::infinity();
    return Vector<pair<double, double> >(D, make_pair(-inf, inf));
}

template<typename FUNCTION, typename M_SAMPLER> pair<Vector<double>,
    double> markovianMinimize(FUNCTION const& f, M_SAMPLER const& s,
    Vector<double> x0, int maxEvals = 1000000)
{
    assert(maxEvals > 0 && isfinite(norm(x0)));
    pair<Vector<double>, double> xy(x0, f(x0));
    while(--maxEvals > 0)
    {
        Vector<double> xNext = s(xy.first, xy.first);
        assert(isfinite(norm(xNext)));
        double yNext = f(xNext);
        if(isnan(xy.second) || yNext < xy.second)
        {
            xy.first = xNext;
            xy.second = yNext;
        }
    }
    return xy;
}

template<typename FUNCTION, typename SAMPLER, typename M_SAMPLER>
pair<Vector<double>, double> smallBudgetHybridMinimize(FUNCTION const& f,
    SAMPLER const& s, M_SAMPLER const& ms, int maxEvals = 100)
{
    assert(maxEvals > 2);//no point to have fewer
    return markovianMinimize(f, ms,
        RBCDGeneral(f, s, s(), maxEvals * 0.5).first, maxEvals * 0.5);
}

template<typename FUNCTION, typename SAMPLER, typename M_SAMPLER>
pair<Vector<double>, double> hybridBeforeLocalMinimize(FUNCTION const& f,
    SAMPLER const& s, M_SAMPLER const& ms, int maxEvals = 1000000)
{
    assert(maxEvals > 100);//no point to have fewer
    return hybridLocalMinimize(smallBudgetHybridMinimize(f, s, ms,
        maxEvals * 0.9).first, f, maxEvals * 0.1);
}

template<typename FUNCTION, typename SAMPLER> pair<Vector<double>, double>
    differentialEvolutionMinimize(FUNCTION const& f, SAMPLER const& s,
    Vector<pair<double, double> > const& box, int maxEvals = 1000000)
{
    assert(maxEvals > 0);
    int n = pow(maxEvals, 1.0/3);
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
            Vector<int> jkl = GlobalRNG().randomCombination(3, n);
            Vector<double> xiNew = population[i].first,
                xiMutated = boxTrim(population[jkl[0]].first +
                population[jkl[1]].first - population[jkl[2]].first, box);
            //crossover with mutated point
            int D = xiNew.getSize(), randK = GlobalRNG().mod(D);
            for(int k = 0; k < D; ++k) if(GlobalRNG().mod(2) || k == randK)
                xiNew[k] = xiMutated[k];
            if(!isfinite(norm(xiNew)))
            {//enforce finite samples
                ++maxEvals;
                continue;
            }
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

template<typename FUNCTION, typename SAMPLER, typename M_SAMPLER>
pair<Vector<double>, double> hybridExplorativeBeforeLocalMinimize(
    FUNCTION const& f, SAMPLER const& s, M_SAMPLER const& ms,
    Vector<pair<double, double> > const& box, int maxEvals = 1000000,
    double yPrecision = highPrecEps)
{
    assert(maxEvals > 100);//no point to have fewer
    return hybridLocalMinimize(markovianMinimize(f, ms,
        simulatedAnnealingMinimize(f, RBCDGeneral(f, s,
        differentialEvolutionMinimize(f, s, box, maxEvals * 0.22).first,
        maxEvals * 0.22).first, ms, maxEvals * 0.22).first,
        maxEvals * 0.22).first, f, maxEvals * 0.12, yPrecision);
}

//presented in numerical algs
template<typename FUNCTION> struct NormFunction
{
    FUNCTION f;
    NormFunction(FUNCTION const& theF): f(theF){}
    double operator()(Vector<double> const& x)const{return norm(f(x));}
};
template<typename FUNCTION> pair<Vector<double>, double> solveByOptimization(
    FUNCTION const& f, Vector<double> const& x0, int maxEvals = 1000000)
{//use scaled grad as error estimate
    NormFunction<FUNCTION> nf(f);
    pair<Vector<double>, double> xy = hybridBeforeLocalMinimize(nf,
        UnboundedSampler(x0), AgnosticStepSampler(), maxEvals - 2);
    double errorEstimate = max(normInf(estimateGradientCD(xy.first, nf))/
        max(1.0, abs(xy.second)), defaultPrecEps);
    return make_pair(xy.first, errorEstimate);
}

template<typename FUNCTION> pair<Vector<double>, double>
hybridEquationSolve(FUNCTION const& f, int D,
    double xEps = highPrecEps, int maxEvals = 1000000)
{//first opt, then Broyden local search on opt result to improve precision
    int broydenEvals = max(1000, maxEvals/4),
        optEvals = maxEvals/2 - broydenEvals;
    pair<Vector<double>, double> result = solveByOptimization(f,
        Vector<double>(D), optEvals), result2 =
        solveBroydenHybrid(f, result.first, xEps, broydenEvals);
    //keep opt error estimate if Broyden did nothing
    if(!isfinite(result2.second)) result2.second = result.second;
    //do random Broyden with the other half of evals
    result = solveBroydenLevy(f, D, xEps, maxEvals/2/1000);
    if(normInf(f(result.first)) < normInf(f(result2.first)))
        result = result2;
    return result;
}

}//end namespace igmdk
#endif
