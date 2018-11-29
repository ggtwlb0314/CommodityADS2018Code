#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <limits>
#include <memory> //for shared ptr
#include "NumericalMethods.h"
#include "NumericalOptimization.h"
#include "../RandomNumberGeneration/Statistics.h"
#include "../Utils/DEBUG.h"
#include "../ExternalMemoryAlgorithms/CSV.h"
#include "NumericalMethodsTestAuto.h"
using namespace std;
using namespace igmdk;

int evalCount2 = 0;



struct TestFunctions1D
{//for integral use http://www.emathhelp.net/calculators/calculus-2/definite-integral-calculator
    struct BaseF
    {
        virtual double operator()(double x)const = 0;
        virtual string name()const{return "";}
        virtual double getA()const{return -1;}
        virtual double getB()const{return 1;}
        virtual double deriv(double x)const//not needed in most cases
            {return numeric_limits<double>::quiet_NaN();}
        virtual double integralAB()const//not needed in most cases
            {return numeric_limits<double>::quiet_NaN();}
    };
    struct Step: public BaseF//discontinuity
    {
        double operator()(double x)const{return x > 0;}
        string name()const{return "Step";}
        double deriv(double x)const{return 0;}
        double integralAB()const{return 1;}
    };
    struct Abs: public BaseF//non-differentiable
    {
        double operator()(double x)const{return abs(x);}
        string name()const{return "Abs";}
        double deriv(double x)const{return x = 0 ? 0 : x < 0 ? -1 : 1;}
        double integralAB()const{return 1;}
    };
    struct Lin: public BaseF//exact for order 1
    {
        double operator()(double x)const{return x;}
        string name()const{return "Lin";}
        double deriv(double x)const{return 1;}
        double integralAB()const{return 0;}
    };
    struct Square: public BaseF//exact for order 2
    {
        double operator()(double x)const{return x * x;}
        string name()const{return "Square";}
        double deriv(double x)const{return 2 * x;}
        double integralAB()const{return 2.0/3;}
    };
    struct Cube: public BaseF//exact for order 3
    {
        double operator()(double x)const{return x * x * x;}
        string name()const{return "Cube";}
        double deriv(double x)const{return 3 * x * x;}
        double integralAB()const{return 0;}
    };
    struct Quad: public BaseF//exact for order 2
    {
        double operator()(double x)const{return x * x * x * x;}
        string name()const{return "Quad";}
        double deriv(double x)const{return 4 * x * x * x;}
        double integralAB()const{return 0.4;}
    };
    struct Exp: public BaseF//analytic
    {
        double operator()(double x)const{return exp(x);}
        string name()const{return "Exp";}
        double deriv(double x)const{return exp(x);}
        double integralAB()const
        {
            double e = exp(1);
            return e - 1/e;
        }
    };
    struct SqrtAbs: public BaseF
    {
        double operator()(double x)const{return sqrt(abs(x));}
        string name()const{return "SqrtAbs";}
        //NEED DERIV!
        double integralAB()const{return 4.0/3;}
    };
    struct Runge: public BaseF//differentiable nonlinear
    {//deriv not given - too complex
        double operator()(double x)const{return 1/(1 + 25 * x * x);}
        string name()const{return "Runge";}
        double getA()const{return -5;}
        double getB()const{return 5;}
        double integralAB()const{return 0.4 * atan(25);}
    };
    struct Log: public BaseF//analytic slow growth
    {
        double operator()(double x)const{return log(x);}
        string name()const{return "Log";}
        double deriv(double x)const{return 1/x;}
        double getA()const{return 1;}
        double getB()const{return 2;}
        double integralAB()const{return 2 * log(2) - 1;}
    };
    struct XSinXM1: public BaseF//continuous not Lipschitz
    {
        double operator()(double x)const
        {
            double temp = x * sin(1/x);
            return isfinite(temp) ? temp : 0;
        }
        string name()const{return "XSinXM1";}
        double deriv(double x)const
        {
            double temp = 2 * x * sin(1/x) - cos(1/x);
            return isfinite(temp) ? temp : 0;
        }
        //exact integrator struggles here - use 0.00000000000001 to 1
        double integralAB()const{return 2 * 0.378530017124161;}
    };
    struct Sin: public BaseF//periodic analytic
    {
        double operator()(double x)const{return sin(x);}
        string name()const{return "Sin";}
        double deriv(double x)const{return cos(x);}
        double integralAB()const{return 0;}
    };
    struct AbsIntegral: public BaseF//single continuous derivative
    {
        double operator()(double x)const{return (x > 0 ? 1 : -1) * x * x/2;}
        string name()const{return "AbsIntegral";}
        double deriv(double x)const{return abs(x);}
        double integralAB()const{return 0;}
    };
    struct Tanh: public BaseF//bounded range slow growth
    {
        double operator()(double x)const{return tanh(x);}
        string name()const{return "Tanh";}
        double deriv(double x)const{return 1 - tanh(x) * tanh(x);}
        double integralAB()const{return 0;}
    };
    struct NormalPDF: public BaseF//analytic thin tails
    {
        double operator()(double x)const{return exp(-x*x/2)/sqrt(2 * PI());}
        string name()const{return "NormalPDF";}
        double deriv(double x)const{return -x * (*this)(x);}
        double getA()const{return -10;}
        double getB()const{return 10;}
        double integralAB()const{return 1;}
    };
    struct DeltaPDF: public BaseF//non-diff continuous, 0 tails
    {
        double operator()(double x)const{return x > -1 && x < 1 ? 1 - abs(x) : 0;}
        string name()const{return "DeltaPDF";}
        double deriv(double x)const{return x > -1 && x < 1 ? -(x = 0 ? 0 : x < 0 ? -1 : 1) : 0;}
        double getA()const{return -20;}
        double getB()const{return 20;}
        double integralAB()const{return 1;}
    };
    struct Kahaner21: public BaseF//hard to notice bump
    {
        double operator()(double x)const{return 1/pow(cosh(10 * x - 2), 2) +
            1/pow(cosh(100 * x - 40), 4) + 1/pow(cosh(1000 * x - 600), 6);}
        string name()const{return "Kahaner21";}
        double getA()const{return 0;}
        double integralAB()const{return 0.210802735500549277;}
    };
    struct F575: public BaseF//singularity at 0
    {
        struct F575Helper
        {//singularity at 0
            double operator()(double x)const
                {return exp(-0.4 * x) * cos(2 * x)/pow(x, 0.7);}
        };
        SingularityWrapper<F575Helper> fh;
        double operator()(double x)const{return fh(x);}
        string name()const{return "F575";}
        double getA()const{return 0;}
        double getB()const{return 100;}
        double integralAB()const{return 2.213498276272980295056;}
    };
    static int evalCount;
    struct MetaF
    {
        shared_ptr<BaseF> f;
        template<typename F> MetaF(shared_ptr<F> const& theF): f(theF){}
        double operator()(double x)const
        {
            assert((*f).getA() <= x && x <= (*f).getB());//do nan not assert?
            ++evalCount;
            return (*f)(x);
        }
        double deriv(double x)const
        {
            assert((*f).getA() <= x && x <= (*f).getB());//do nan not assert?
            return f->deriv(x);
        }
        string getName()const{return f->name();}
        double integralAB()const{return f->integralAB();}
        double getA()const{return f->getA();}
        double getB()const{return f->getB();}
    };
    static Vector<MetaF> getFunctions()
    {
        Vector<MetaF> result;
        result.append(MetaF(make_shared<Step>()));
        result.append(MetaF(make_shared<Abs>()));
        result.append(MetaF(make_shared<Lin>()));
        result.append(MetaF(make_shared<Square>()));
        result.append(MetaF(make_shared<Cube>()));
        result.append(MetaF(make_shared<Quad>()));
        result.append(MetaF(make_shared<Exp>()));
        result.append(MetaF(make_shared<SqrtAbs>()));
        result.append(MetaF(make_shared<Runge>()));
        result.append(MetaF(make_shared<Log>()));
        result.append(MetaF(make_shared<XSinXM1>()));
        result.append(MetaF(make_shared<Sin>()));
        result.append(MetaF(make_shared<AbsIntegral>()));
        result.append(MetaF(make_shared<Tanh>()));
        result.append(MetaF(make_shared<NormalPDF>()));
        result.append(MetaF(make_shared<DeltaPDF>()));
        result.append(MetaF(make_shared<F575>()));
        return result;
    }
};
int TestFunctions1D::evalCount = 0;//HANDLE INTERGRATION FUNCS NEXT!

template<typename FUNCTION> struct MCIFunctionHelper
{
    FUNCTION const& f;
    MCIFunctionHelper(FUNCTION const& theF): f(theF){}
    double operator()(Vector<double>const& x)const
    {
        return f(x[0]);
    }
};


void printResult(double result, double answer)
{
    double relResult = answer == 0 ? result : abs(answer - result)/answer;
    DEBUG(result);
    DEBUG(answer);
    DEBUG(relResult);
}
void printResult(pair<double, double> const result, double answer)
{
    double relResult = answer == 0 ? result.first : abs(answer - result.first)/answer;
    DEBUG(relResult);
    DEBUG(result.second);
    DEBUG(evalCount2);
    evalCount2 = 0;
}

void printResult2(pair<double, double> const result, double answer, Vector<Vector<string> > & matrix)
{
    double relResult = abs(answer - result.first)/max(1.0, answer);
    DEBUG(relResult);
    DEBUG(result.second);
    DEBUG(TestFunctions1D::evalCount);
    //matrix.lastItem().append(toString(relResult));
    //matrix.lastItem().append(toString(result.first));
    matrix.lastItem().append(toString(abs(answer - result.first)));
    matrix.lastItem().append(toString(result.second));
    matrix.lastItem().append(toString(TestFunctions1D::evalCount));
    TestFunctions1D::evalCount = 0;
}

template<typename FUNCTION> pair<double, double> integrateCCDoubling(
    FUNCTION const& f, double a, double b, double maxRelAbsError = highPrecEps,
    int maxEvals = 5000, int minEvals = 17)
{//use doubling for error estimate and convergence criteria
    int n = minEvals - 1;
    assert(minEvals <= maxEvals && isPowerOfTwo(n));
    Vector<double> fx(n + 1);
    ScaledFunctionM11<FUNCTION> fM11(a, b, f);
    for(int i = 0; i <= n; ++i) fx[i] = fM11(cos(PI() * i/n));
    ScaledChebAB che(fx, a, b);
    double result = che.integrate().first,//want to double once
        oldResult = numeric_limits<double>::quiet_NaN();
    while(maxEvals >= fx.getSize() + n && (isnan(oldResult) ||
        !isEEqual(result, oldResult, maxRelAbsError)))
    {
        fx = reuseChebEvalPoints(fM11, fx);
        ScaledChebAB che2(fx, a, b);
        n *= 2;
        oldResult = result;
        result = che2.integrate().first;
    }
    return make_pair(result, abs(result - oldResult));
}

class IntervalSimpson
{
    double left, dx, fLeft, fLM, fMiddle, fMR, fRight;
    static double integrateHelper(double f0, double f1, double f2, double dx)
        {return dx * (f0 + 4 * f1 + f2)/3;}
    template<typename FUNCTION> initHelper(FUNCTION const& f, double a,
        double b, double fa, double fm, double fb)
    {
        left = a;
        dx = (b - a)/4;
        fLeft = fa;
        fLM = f(a + dx);
        fMiddle = fm;
        fMR = f(a + 3 * dx);
        fRight = fb;
    }
    template<typename FUNCTION> IntervalSimpson(FUNCTION const& f, double a,
        double b, double fa, double fm, double fb)
            {initHelper(f, a, b, fa, fm, fb);}
public:
    template<typename FUNCTION> IntervalSimpson(FUNCTION const& f, double a,
        double b){initHelper(f, a, b, f(a), f(a + (b - a)/4), f(b));}
    double integrate()const
    {
        return integrateHelper(fLeft, fLM, fMiddle, dx) +
            integrateHelper(fMiddle, fMR, fRight, dx);
    }
    double length()const{return 4 * dx;}
    double error()const
    {
        return abs(integrate() -
            integrateHelper(fLeft, fMiddle, fRight, 2 * dx));
    }
    template<typename FUNCTION>
    Vector<IntervalSimpson> split(FUNCTION const& f)const
    {
        Vector<IntervalSimpson> result;
        double middle = left + 2 * dx, right = middle + 2 * dx;
        result.append(IntervalSimpson(f, left, middle, fLeft, fLM, fMiddle));
        result.append(IntervalSimpson(f, middle, right, fMiddle, fMR, fRight));
        return result;
    }
    static int initEvals(){return 5;}
    static int splitEvals(){return 4;}
};


template<typename FUNCTION> pair<double, double> integrateHybridSimpson(
    FUNCTION const& f, double a, double b, double eRelAbs = highPrecEps,
    int maxEvals = 1000000, int minEvals = -1)
{
    int CCEvals = min(maxEvals/2, 1000);
    pair<double, double> resultCC = integrateCC(f, a, b, CCEvals);
    if(isEEqual(resultCC.first, resultCC.first + resultCC.second, eRelAbs))
        return resultCC;
    pair<double, double> resultAS = integrateAdaptiveHeap<IntervalSimpson>(f,
        a, b, eRelAbs, maxEvals - CCEvals, minEvals);
    return resultCC.second < resultAS.second ? resultCC : resultAS;
}
template<typename FUNCTION> void testIntegrateHelper(FUNCTION const& f, double left, double right,
    Vector<Vector<string> > & matrix)
{
    double answer = f.integralAB();
    int n = 10000000;
    DEBUG("CC");
    matrix.lastItem().append("CC");
    printResult2(integrateCC(f, left, right), answer, matrix);
    DEBUG("CCDoubling");
    matrix.lastItem().append("CCDoubling");
    printResult2(integrateCCDoubling(f, left, right), answer, matrix);
    DEBUG("AdaptSimp");
    matrix.lastItem().append("AS");
    printResult2(integrateAdaptiveHeap<IntervalSimpson>(f, left, right), answer, matrix);
    DEBUG("AdaptGLK");
    matrix.lastItem().append("AGLK");
    printResult2(integrateAdaptiveHeap<IntervalGaussLobattoKronrod>(f, left, right), answer, matrix);
    DEBUG("Hybr");
    matrix.lastItem().append("Hybr");
    printResult2(integrateHybrid(f, left, right), answer, matrix);
    DEBUG("MC");
    matrix.lastItem().append("MC");
    printResult2(MonteCarloIntegrate(Vector<pair<double, double> >(1, make_pair(left, right)),
        n, InsideTrue(), MCIFunctionHelper<FUNCTION>(f)), answer, matrix);
    DEBUG("TrapData");
    matrix.lastItem().append("TrapData");
    Vector<pair<double, double> > evals;
    evals.append(pair<double, double>(left, f(left)));
    evals.append(pair<double, double>(right, f(right)));
    for(int i = 0; i < n - 2; ++i)
    {
        double x = GlobalRNG().uniform(left, right);
        evals.append(pair<double, double>(x, f(x)));
    }
    printResult2(integrateFromData(evals), answer, matrix);
}

void testIntegrators()
{
    Vector<Vector<string> > matrix;
    Vector<TestFunctions1D::MetaF> fs = TestFunctions1D::getFunctions();
    for(int i = 0; i < fs.getSize(); ++i)
    {
        if(!isfinite(fs[i].integralAB())) continue;
        string name = fs[i].getName();
        DEBUG(name);
        matrix.append(Vector<string>());
        matrix.lastItem().append(name);
        testIntegrateHelper(fs[i], fs[i].getA(), fs[i].getB(), matrix);
    }
    int reportNumber = time(0);
    string filename = "reportIntegrate" + toString(reportNumber) + ".csv";
    createCSV(matrix, filename.c_str());
}


template<typename TWO_VAR_FUNCTION>
double EulerSolve(TWO_VAR_FUNCTION const& f, double x0, double xGoal,
    double y0, int nIntervals = 10000)
{
    assert(nIntervals > 0);
    double h = (xGoal - x0)/nIntervals, y = y0;
    for(double x = x0; nIntervals--; x += h) y += h * f(x, y);
    return y;
}
template<typename TWO_VAR_FUNCTION>
pair<double, double> adaptiveRungKutta4(TWO_VAR_FUNCTION const& f, double x0,
    double xGoal, double y0, double localERelAbs = defaultPrecEps,
    int maxIntervals = 100000, int minIntervals = -1,
    double upFactor = pow(2, 0.2))
{
    if(minIntervals == -1) minIntervals = sqrt(maxIntervals);
    assert(xGoal > x0 && minIntervals > 0 && upFactor > 1);
    double hMax = (xGoal - x0)/minIntervals, hMin = (xGoal - x0)/maxIntervals,
        linearError = 0, h1 = hMax, y = y0,
        y1 = numeric_limits<double>::quiet_NaN(), f0;
    bool last = false;
    for(double x = x0; !last;)
    {
        if(x + h1 > xGoal)
        {//make last step accurate
            h1 = xGoal - x;
            last = true;
        }
        if(isnan(y1))
        {
            f0 = f(x, y);
            y1 = RungKutta4Step(f, x, y, h1, f0);
        }
        double h2 = h1/2, y2 = RungKutta4Step(f, x, y, h2, f0), firstY2 = y2,
            xFraction = h1/(xGoal - x0);
        y2 = RungKutta4Step(f, x + h2, y2, h2);
        if(h2 < hMin || isEEqual(y2, y1,
            max(highPrecEps, localERelAbs * sqrt(xFraction))))
        {//accept step
            x += h1;
            y = y2;
            linearError += abs(y2 - y1);
            y1 = numeric_limits<double>::quiet_NaN();
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

template<typename FUNCTION> struct DummyODE1D
{
    FUNCTION f;
    double operator()(double x, double y)const{return f(x);}
};
template<typename FUNCTION> void testODEHelper(FUNCTION const& f, double left, double right,
    Vector<Vector<string> > & matrix)
{
    DummyODE1D<FUNCTION> fxy = {f};
    double exact = f.integralAB();
    DEBUG("Euler 10k");
    matrix.lastItem().append("Euler 10k");
    printResult2(make_pair(EulerSolve(fxy, left, right, 0), 0), exact, matrix);
    DEBUG("doublingRungKutta4");
    matrix.lastItem().append("doublingRungKutta4");
    printResult2(adaptiveRungKutta4(fxy, left, right, 0), exact, matrix);
    DEBUG("DormandPrice");
    matrix.lastItem().append("DormandPrice");
    printResult2(adaptiveRungKuttaDormandPrice(fxy, left, right, 0), exact, matrix);
}

void testODE()
{
    Vector<Vector<string> > matrix;
    Vector<TestFunctions1D::MetaF> fs = TestFunctions1D::getFunctions();
    for(int i = 0; i < fs.getSize(); ++i)
    {
        if(!isfinite(fs[i].integralAB())) continue;
        string name = fs[i].getName();
        DEBUG(name);
        matrix.append(Vector<string>());
        matrix.lastItem().append(name);
        testODEHelper(fs[i], fs[i].getA(), fs[i].getB(), matrix);
    }
    int reportNumber = time(0);
    string filename = "reportODE" + toString(reportNumber) + ".csv";
    createCSV(matrix, filename.c_str());
}

struct D2F
{
    double operator()(double x, double y)const{++evalCount2; return x + y;}
};
//bad trap example
struct D2F2
{
    double operator()(double x, double y)const{++evalCount2; return x == 0 ? 1 : y * (-2 * x + 1/x);}
};
void testRungKutta()
{
    DEBUG("Adaptive2");
    double x = 1;
    double exact = 3 * exp(x) - x - 1;
    DEBUG(exact);
    printResult(adaptiveRungKutta4(D2F(), 0, 1, 2), exact);
    x = 2;
    exact = x * exp(-x * x);
    DEBUG(exact);
    printResult(adaptiveRungKutta4(D2F2(), 0, 2, 0), exact);
    DEBUG("AdaptiveDP");
    x = 1;
    exact = 3 * exp(x) - x - 1;
    DEBUG(exact);
    printResult(adaptiveRungKuttaDormandPrice(D2F(), 0, 1, 2), exact);
    x = 2;
    exact = x * exp(-x * x);
    DEBUG(exact);
    printResult(adaptiveRungKuttaDormandPrice(D2F2(), 0, 2, 0), exact);
    DEBUG("E,MM");
    x = 1;
    exact = 3 * exp(x) - x - 1;
    DEBUG(exact);
    printResult(make_pair(EulerSolve(D2F(), 0, 1, 2), 0), exact);
    x = 2;
    exact = x * exp(-x * x);
    DEBUG(exact);
    printResult(make_pair(EulerSolve(D2F2(), 0, 2, 0), 0), exact);
}

struct TestFunctionsMultiD
{//for integral use http://www.emathhelp.net/calculators/calculus-2/definite-integral-calculator
    struct BaseF
    {
        virtual Vector<double> operator()(Vector<double> const& x)const = 0;
        virtual string name()const = 0;
        virtual Vector<double> getX0()const = 0;
        virtual Vector<double> getAnswer()const = 0;
    };
    struct TestFunc0
    {//from Fausett
        struct Func1 : public MultivarFuncHelper::F1DBase
        {
            double operator()(Vector<double> const& x)const
                {return x[0] + pow(x[0], 3) + 10 * x[0] - x[1] - 5;}
        };
        struct Func2 : public MultivarFuncHelper::F1DBase
        {
            double operator()(Vector<double> const& x)const
                {return x[1] + x[0] + pow(x[1], 3) - 10 * x[1] + 1;}
        };
        struct Func: public BaseF
        {
            MultivarFuncHelper f;
            Func1 f1;
            Func2 f2;
            Func()
            {
                f.fs.append(&f1);
                f.fs.append(&f2);
            }
            Vector<double> operator()(Vector<double> const& x)const{return f(x);}
            string name()const{return "TestFunc0";}
            Vector<double> getX0()const{return Vector<double>(2, 0.6);}
            Vector<double> getAnswer()const{return Vector<double>(2, 0);}//???
        };
    };
    struct TestFunc1
    {//from Fausett
        struct Func1 : public MultivarFuncHelper::F1DBase
        {
            double operator()(Vector<double> const& x)const
                {return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] - 1;}
        };
        struct Func2 : public MultivarFuncHelper::F1DBase
        {
            double operator()(Vector<double> const& x)const
                {return x[0] * x[0] + x[2] * x[2] - 0.25;}
        };
        struct Func3 : public MultivarFuncHelper::F1DBase
        {
            double operator()(Vector<double> const& x)const
                {return x[0] * x[0] + x[1] * x[1] - 4 * x[2];}
        };
        struct Func: public BaseF
        {
            MultivarFuncHelper f;
            Func1 f1;
            Func2 f2;
            Func3 f3;
            Func()
            {
                f.fs.append(&f1);
                f.fs.append(&f2);
                f.fs.append(&f3);
            }
            Vector<double> operator()(Vector<double> const& x)const{return f(x);}
            string name()const{return "TestFunc1";}
            Vector<double> getX0()const{return Vector<double>(3, 1);}
            Vector<double> getAnswer()const{return Vector<double>(3, 0);}//???
        };
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
            }
            return x0;
        }
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
    };
    struct Trig: public BaseF
    {//From Dennis & Schnabel, they give no solution, but x = 0 works
        int n;
        Trig(int theN = 2): n(theN) {}
        Vector<double> operator()(Vector<double> const& x)const
        {

            double cSum = 0;
            for(int i = 0; i < n; ++i) cSum += cos(x[i]);
            Vector<double> fx(n, n - cSum);
            for(int i = 0; i < n; ++i)
                fx[i] += (i + 1) * (1 - cos(x[i])) - sin(x[i]);
            return fx;
        }
        string name()const{return "Trig" + toString(n);}
        Vector<double> getX0()const{return Vector<double>(n, 1.0/n);}
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
    struct Wood: public BaseF
    {//From Dennis & Schnabel; More et al have different definition
        Vector<double> operator()(Vector<double> const& x)const
        {
            Vector<double> fx(4);
            double t0 = x[0] * x[0] - x[1], t01 = 1 - x[0],
                t1 = x[2] * x[2] - x[3], t11 = 1 - x[2],
                t2 = 1 - x[1], t3 = 1 - x[3];
            fx[0] = 100 * t0 * t0 + t01 * t01;
            fx[1] = 90 * t1 * t1 + t11 * t11;
            fx[2] = 10.1 * (t2 * t2 + t3 * t3);
            fx[3] = 19.8 * t2 * t3;
            return fx;
        }
        string name()const{return "Wood";}
        Vector<double> getX0()const
        {
            Vector<double> x0(4, 0);
            x0[0] = -3;
            x0[1] = -1;
            x0[2] = -3;
            x0[3] = -1;
            return x0;
        }
        Vector<double> getAnswer()const{return Vector<double>(4, 1);}
    };
    struct MultiArctan: public BaseF
    {//swings Newton without backtrack into inf per Kelley
        int n;
        MultiArctan(int theN = 2): n(theN) {}
        Vector<double> operator()(Vector<double> const& x)const
        {
            Vector<double> fx(n);
            for(int i = 0; i < n; ++i) fx[i] = atan(x[i]);
            return fx;
        }
        string name()const{return "MultiArctan" + toString(n);}
        Vector<double> getX0()const{return Vector<double>(n, 10);}
        Vector<double> getAnswer()const{return Vector<double>(n, 0);}
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
    struct GulfRND: public BaseF
    {//From More et al; paper has mistake "mi" should be "-" not m * (i + 1)
        Vector<double> operator()(Vector<double> const& x)const
        {
            Vector<double> fx(3);
            for(int i = 0; i < fx.getSize(); ++i)
            {
                double ti = (i + 1.0)/100, yi = 25 + pow(-50 * log(ti), 2.0/3);
                fx[i] = exp(-pow(abs(yi - x[1]), x[2])/x[0]) - ti;
            }
            return fx;
        }
        string name()const{return "GulfRND";}
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
        Vector<double> operator()(Vector<double> const& x)const
        {
            ++evalCount;
            return (*f)(x);
        }
        string getName()const{return f->name();}
        virtual Vector<double> getX0()const{return f->getX0();}
        virtual Vector<double> getAnswer()const{return f->getAnswer();}
    };
    static Vector<MetaF> getFunctions()
    {
        Vector<MetaF> result;
        //result.append(MetaF(make_shared<TestFunc0::Func>()));
        //result.append(MetaF(make_shared<TestFunc1::Func>()));
        result.append(MetaF(make_shared<ExtendedRosenbrock>()));
        result.append(MetaF(make_shared<ExtendedPowellSingular>()));
        result.append(MetaF(make_shared<Trig>()));
        result.append(MetaF(make_shared<MultiArctan>()));
        result.append(MetaF(make_shared<HelicalValley>()));
        result.append(MetaF(make_shared<LinearFFullRank>()));
        result.append(MetaF(make_shared<Wood>()));
        result.append(MetaF(make_shared<GulfRND>()));
        result.append(MetaF(make_shared<BiggsExp6>()));
        result.append(MetaF(make_shared<ExtendedRosenbrock>(10)));
        result.append(MetaF(make_shared<ExtendedPowellSingular>(12)));
        result.append(MetaF(make_shared<Trig>(10)));
        result.append(MetaF(make_shared<MultiArctan>(10)));
        result.append(MetaF(make_shared<LinearFFullRank>(10)));
        result.append(MetaF(make_shared<ExtendedRosenbrock>(100)));
        result.append(MetaF(make_shared<ExtendedPowellSingular>(100)));
        result.append(MetaF(make_shared<Trig>(100)));
        result.append(MetaF(make_shared<MultiArctan>(100)));
        result.append(MetaF(make_shared<LinearFFullRank>(100)));
        return result;
    }
};
int TestFunctionsMultiD::evalCount = 0;

template<typename FUNCTION>
void printResultNonliearEq(pair<Vector<double>, double> const& result,
    FUNCTION const& f, Vector<Vector<string> >& matrix)
{
    int evalCount = TestFunctionsMultiD::evalCount;
    DEBUG(normInf(result.first - f.getAnswer()));
    matrix.lastItem().append(toString(normInf(result.first - f.getAnswer())));
    DEBUG(result.second);
    matrix.lastItem().append(toString(result.second));
    DEBUG(normInf(f(result.first)));
    matrix.lastItem().append(toString(normInf(f(result.first))));
    DEBUG(evalCount);
    matrix.lastItem().append(toString(evalCount));
    TestFunctionsMultiD::evalCount = 0;
}
template<typename FUNCTION> void testNonlinearEqHelper(FUNCTION const& f,
    Vector<double> const& x0, Vector<Vector<string> >& matrix)
{
    DEBUG("Broyden QR");
    matrix.lastItem().append("Broyden QR");
    printResultNonliearEq(solveBroyden(f, x0), f, matrix);
    DEBUG("LMBroyden");
    matrix.lastItem().append("LMBroyden");
    printResultNonliearEq(solveLMBroyden(f, x0), f, matrix);
    DEBUG("BroydenLevy");
    matrix.lastItem().append("BroydenLevy");
    printResultNonliearEq(solveBroydenLevy(f, x0.getSize()), f, matrix);
    DEBUG("Opt");
    matrix.lastItem().append("Opt");
    printResultNonliearEq(solveByOptimization(f, x0), f, matrix);
    DEBUG("Hybrid");
    matrix.lastItem().append("Hybrid");
    printResultNonliearEq(hybridEquationSolve(f, x0.getSize()), f, matrix);
}

void testNonlinearEq()
{
    Vector<Vector<string> > matrix;
    Vector<TestFunctionsMultiD::MetaF> fs = TestFunctionsMultiD::getFunctions();
    for(int i = 0; i < fs.getSize(); ++i)
    {
        string name = fs[i].getName();
        DEBUG(name);
        matrix.append(Vector<string>());
        matrix.lastItem().append(name);
        testNonlinearEqHelper(fs[i], fs[i].getX0(), matrix);
    }
    int reportNumber = time(0);
    string filename = "reportNonlinearSolve" + toString(reportNumber) + ".csv";
    createCSV(matrix, filename.c_str());
}

//from Fausett; ignore x
struct FPStiff1 : public MultivarFuncHelper::F1DBase
{
    double operator()(Vector<double> const& yx)const{++evalCount2; return 98 * yx[0] + 198 * yx[1];}
};
struct FPStiff2 : public MultivarFuncHelper::F1DBase
{
    double operator()(Vector<double> const& yx)const{return -99 * yx[0] - 199 * yx[1];}
};
struct FPStiffDummy : public MultivarFuncHelper::F1DBase
{
    double operator()(Vector<double> const& yx)const{return 0;}
};


template<typename YX_FUNCTION> struct BackwardEulerFunction
{//for Euler don't need f basis and y is fine
    Vector<double> y;
    double x, h;
    YX_FUNCTION f;
    Vector<double> operator()(Vector<double> const& yNext)const
        {return yNext - (y + evalYX(x + h, yNext, f) * h);}
};
struct BackwardEulerStepF
{
    template<typename YX_FUNCTION> Vector<double> operator()(
        YX_FUNCTION const& f, double x, Vector<double> const& y, double h,
        double solveERelAbs)const
    {
        BackwardEulerFunction<YX_FUNCTION> bef = {y, x, h, f};
        return solveBroydenHybrid(bef, y, solveERelAbs).first;
    }
};

template<typename YX_FUNCTION> struct ImplicitTrapezoidFunction
{
    Vector<double> y, fx;
    double x, h;
    YX_FUNCTION f;
    Vector<double> operator()(Vector<double> const& fSumNext)const
        {return fSumNext - (fx + evalYX(x + h, y + fSumNext * h, f)) * 0.5;}
};
struct ImplicitTrapezoidStepF
{
    template<typename YX_FUNCTION> Vector<double> operator()(
        YX_FUNCTION const& f, double x, Vector<double> const& y, double h,
        double solveERelAbs)const
    {//ignore f0 reuse for simplicity
        ImplicitTrapezoidFunction<YX_FUNCTION> itf =
            {y, evalYX(x, y ,f), x, h, f};
        return y + solveBroydenHybrid(itf, Vector<double>(y.getSize(), 0),
            max(solveERelAbs, numeric_limits<double>::epsilon() * normInf(y)/h)
            ).first * h;
    }
};
template<typename YX_FUNCTION, typename STEPF> Vector<double>
    ImplicitFixedStepper(YX_FUNCTION const& f, STEPF const& s, double x0,
    double xGoal, Vector<double> y, int nIntervals = 100000)
{
    assert(nIntervals > 0);
    double h = (xGoal - x0)/nIntervals;
    for(double x = x0; nIntervals--; x += h) y = s(f, x, y, h, defaultPrecEps);
    return y;
}

void testStiffSolvers()
{
    MultivarFuncHelper f;
    FPStiff1 f1;
    FPStiff2 f2;
    FPStiffDummy fd;
    f.fs.append(&f1);
    f.fs.append(&f2);
    f.fs.append(&fd);
    Vector<double> y0(2);
    y0[0] = 1;
    y0[1] = 0;
    DEBUG("Imp Euler");
    Vector<double> result = ImplicitFixedStepper(f, BackwardEulerStepF(), 0, 1, y0);
    DEBUG("result");
    result.debug();
    Vector<double> correctResult(2);
    correctResult[0] = 2 * exp(-1) - exp(-100);
    correctResult[1] = -exp(-1) + exp(-100);
    DEBUG("correctResult");
    correctResult.debug();
    DEBUG(normInf(result - correctResult));
    DEBUG(evalCount2);
    evalCount2 = 0;
    DEBUG("Imp Trap");
    result = ImplicitFixedStepper(f, ImplicitTrapezoidStepF(), 0, 1, y0);
    DEBUG("result");
    result.debug();
    DEBUG(normInf(result - correctResult));
    DEBUG(evalCount2);
    evalCount2 = 0;
    pair<Vector<double>, double> result2;
    DEBUG("Adap Imp Trap");
    result2 = adaptiveStepper(f, ImplicitTrapezoidStepF(), 0, 1, y0);
    result = result2.first;
    DEBUG("result");
    result.debug();
    DEBUG(result2.second);
    DEBUG(normInf(result - correctResult));
    DEBUG(evalCount2);
    evalCount2 = 0;
    DEBUG("RadauIIA5");
    result = ImplicitFixedStepper(f, RadauIIA5StepF(), 0, 1, y0, 100);
    DEBUG("result");
    result.debug();
    DEBUG(normInf(result - correctResult));
    DEBUG(evalCount2);
    evalCount2 = 0;
    DEBUG("Adap RadauIIA5");
    result2 = adaptiveStepper(f, RadauIIA5StepF(), 0, 1, y0);
    result = result2.first;
    DEBUG("result");
    result.debug();
    DEBUG(result2.second);
    DEBUG(normInf(result - correctResult));
    DEBUG(evalCount2);
    evalCount2 = 0;
}

struct DerivatorFD
{
    template<typename FUNCTION>
    double operator()(FUNCTION const& f, double x)const
        {return estimateDerivativeFD(f, x, f(x));}
};
struct DerivatorCD
{
    template<typename FUNCTION>
    double operator()(FUNCTION const& f, double x)const
        {return estimateDerivativeCD(f, x);}
};
template<typename FUNCTION> double estimateDerivativeCD4(FUNCTION const& f,
    double x, double fEFactor = numeric_limits<double>::epsilon())
{
    double h = pow(fEFactor, 1.0/5) * max(1.0, abs(x));
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h))/(12 * h);
}
struct DerivatorCD4
{
    template<typename FUNCTION>
    double operator()(FUNCTION const& f, double x)const
        {return estimateDerivativeCD4(f, x);}
};
struct DerivatorCheb
{
    ScaledChebAB pCheb;
    DerivatorCheb(ScaledChebAB const& thePCheb): pCheb(thePCheb) {}
    template<typename FUNCTION> double operator()(FUNCTION const& dummy, double x)const
        {return pCheb.evalDeriv(x);}
};
struct DerivatorChebPiecewise
{
    typedef GenericPiecewiseInterpolation<IntervalCheb::INTERPOLANT> T;
    T pCheb;
    DerivatorChebPiecewise(T const& thePCheb): pCheb(thePCheb) {}
    template<typename FUNCTION> double operator()(FUNCTION const& dummy, double x)const
        {return pCheb.evalDeriv(x);}
};

template<typename FUNCTION> void testDerivHelper(FUNCTION const& f, double a, double b,
    Vector<Vector<string> > & matrix)
{
    double length = b - a, safety = length * 0.05;
    a += safety;
    b -= safety;

    DEBUG("FD");
    matrix.lastItem().append("FD");
    testDerivator(f, DerivatorFD(), a, b, matrix);
    DEBUG("CD");
    matrix.lastItem().append("CD");
    testDerivator(f, DerivatorCD(), a, b, matrix);
    DEBUG("CD4");
    matrix.lastItem().append("CD4");
    testDerivator(f, DerivatorCD4(), a, b, matrix);
    DEBUG("Cheb Doubling");
    matrix.lastItem().append("Cheb Doubling");
    DerivatorCheb dc(ScaledChebAB(adaptiveChebEstimate(ScaledFunctionM11<FUNCTION>(a, b, f)), a, b));
    testDerivator(f, dc, a, b, matrix);
    DEBUG("Cheb64 Adaptive Piecewise");
    matrix.lastItem().append("Cheb64 Adaptive Piecewise");
    DerivatorChebPiecewise dcp(interpolateAdaptiveHeap<IntervalCheb>(f, a, b, 64).first);
    testDerivator(f, dcp, a, b, matrix);
    //change below to just use 3 or 4 points
    //check spline also for given data
    /*DEBUG("Bary Adaptive Piecewise");
    matrix.lastItem().append("Bary 3 Adaptive Piecewise");
    DerivatorBaryPiecewise dbp(interpolateAdaptiveHeap<IntervalBarycentric>(f, a, b, 3).first);
    testDerivator(f, dbp, a, b, matrix);
    DEBUG("Bary Adaptive Piecewise Matrix");
    matrix.lastItem().append("Bary 3 Adaptive Piecewise Matrix");
    DerivatorBaryPiecewiseMatrix dbpm(interpolateAdaptiveHeap<IntervalBarycentric>(f, a, b, 3).first.deriver());
    testDerivator(f, dbpm, a, b, matrix);*/
}

template<typename FUNCTION, typename DERIVATOR> pair<double, double> testDerivator(
    FUNCTION const& f, DERIVATOR const& de, double a, double b,
    Vector<Vector<string> > & matrix, int n = 1000000)
{
    DEBUG("eval start");
    double maxRandErr = 0, l2 = 0;
    for(int i = 0; i < n; ++i)
    {
        double x = GlobalRNG().uniform(a, b), diff = de(f, x) - f.deriv(x);
        maxRandErr = max(maxRandErr, abs(diff));
        l2 += diff * diff;
        if(i == 0)
        {
            DEBUG(TestFunctions1D::evalCount);
            matrix.lastItem().append(toString(TestFunctions1D::evalCount));
        }
    }
    DEBUG(maxRandErr);
    matrix.lastItem().append(toString(maxRandErr));
    DEBUG(sqrt(l2/n));
    TestFunctions1D::evalCount = 0;
}

void testDeriv()
{
    Vector<Vector<string> > matrix;
    Vector<TestFunctions1D::MetaF> fs = TestFunctions1D::getFunctions();
    for(int i = 0; i < fs.getSize(); ++i)
    {
        if(!isfinite(fs[i].deriv((fs[i].getA() + fs[i].getB())/2))) continue;
        string name = fs[i].getName();
        DEBUG(name);
        matrix.append(Vector<string>());
        matrix.lastItem().append(name);
        testDerivHelper(fs[i], fs[i].getA(), fs[i].getB(), matrix);
    }
    int reportNumber = time(0);
    string filename = "reportDeriv" + toString(reportNumber) + ".csv";
    createCSV(matrix, filename.c_str());
}

void testAllRootSolver()
{
    Vector<double> coefs(2);//x^2 - 1 = 0, roots: 1, -1
    coefs[0] = -1;
    coefs[1] = 0;
    Vector<complex<double> > roots = findAllRoots(coefs);
    DEBUG("roots");
    roots.debug();
    Vector<double> coefs2(3);//MATLAB EXAMPLE x^3 -7x + 6 = 0, roots = 1, 2, 3
    coefs2[0] = 6;
    coefs2[1] = -7;
    coefs2[2] = 0;
    Vector<complex<double> > roots2 = findAllRoots(coefs2);
    DEBUG("roots2");
    roots2.debug();
    Vector<double> coefs3(3);//0 EXAMPLE x^3 = 0
    coefs3[0] = 0;
    coefs3[1] = 0;
    coefs3[2] = 0;
    Vector<complex<double> > roots3 = findAllRoots(coefs3);
    DEBUG("roots3");
    roots3.debug();
}

template<typename FUNCTION, typename INTERPOLANT> pair<double, double> testInterpolant(
    FUNCTION const& f, INTERPOLANT const& in, double a, double b,
    Vector<Vector<string> > & matrix, int n = 1000000)
{
    DEBUG("eval start");
    DEBUG(TestFunctions1D::evalCount);
    matrix.lastItem().append(toString(TestFunctions1D::evalCount));
    double maxRandErr = 0, l2 = 0;
    for(int i = 0; i < n; ++i)
    {
        double x = GlobalRNG().uniform(a, b), diff = in(x) - f(x);
        maxRandErr = max(maxRandErr, abs(diff));
        l2 += diff * diff;
    }
    DEBUG(maxRandErr);
    matrix.lastItem().append(toString(maxRandErr));
    DEBUG(sqrt(l2/n));
    TestFunctions1D::evalCount = 0;
}


template<typename FUNCTION>
void testInterpolGivenPointsHelper(FUNCTION const& f, Vector<pair<double, double> > const& xy,
    Vector<Vector<string> > & matrix)
{
    assert(xy.getSize() >= 2);
    double a = xy[0].first, b = xy.lastItem().first;
    DEBUG("DynLin");
    DynamicLinearInterpolation dl(xy);
    matrix.lastItem().append("DynLin");
    testInterpolant(f, dl, a, b, matrix);
    DEBUG("NAK Cub");
    NotAKnotCubicSplineInterpolation no(xy);
    matrix.lastItem().append("NAK Cub");
    testInterpolant(f, no, a, b, matrix);
}

template<typename FUNCTION> void testInterpolRandomPointsHelper(FUNCTION const& f, double a, double b,
    Vector<Vector<string> > & matrix, int n = 1000)
{
    DEBUG("Random");
    Vector<pair<double, double> > xy;
    xy.append(make_pair(a, f(a)));
    xy.append(make_pair(b, f(b)));
    n -= 2;
    for(int i = 0; i < n; ++i)
    {
        double x = GlobalRNG().uniform(a, b);
        xy.append(make_pair(x, f(x)));
    }
    quickSort(xy.getArray(), 0, xy.getSize() - 1, PairFirstComparator<double, double>());
    testInterpolGivenPointsHelper(f, xy, matrix);
}

template<typename FUNCTION> void testInterpolChosenPointsHelper(FUNCTION const& f, double a, double b,
    Vector<Vector<string> > & matrix)
{
    DEBUG("Dynamic");
    DEBUG("Cheb Adapt");
    ScaledChebAB cfa(adaptiveChebEstimate(ScaledFunctionM11<FUNCTION>(a, b, f)), a, b);
    matrix.lastItem().append("Cheb Adapt");
    testInterpolant(f, cfa, a, b, matrix);
    DEBUG("Cheb64 Adaptive");
    GenericPiecewiseInterpolation<IntervalCheb::INTERPOLANT> pCheb = interpolateAdaptiveHeap<IntervalCheb>(f, a, b, 64).first;
    matrix.lastItem().append("Cheb64 Adaptive");
    testInterpolant(f, pCheb, a, b, matrix);
}

void testInterpolChosenPoints()
{
    Vector<Vector<string> > matrix;
    bool testDynamic = true, testRandom = true;
    Vector<TestFunctions1D::MetaF> fs = TestFunctions1D::getFunctions();
    for(int i = 0; i < fs.getSize(); ++i)
    {
        string name = fs[i].getName();
        DEBUG(name);
        matrix.append(Vector<string>());
        matrix.lastItem().append(name);
        if(testDynamic) testInterpolChosenPointsHelper(fs[i], fs[i].getA(), fs[i].getB(), matrix);
        if(testRandom) testInterpolRandomPointsHelper(fs[i], fs[i].getA(), fs[i].getB(), matrix);
    }

    int reportNumber = time(0);
    string filename = "reportInterp" + toString(reportNumber) + ".csv";
    DEBUG(matrix.getSize());
    createCSV(matrix, filename.c_str());
    DEBUG(filename);
}
/*
template<typename FUNCTION> void testChebAutoHelper(FUNCTION const& f, double a, double b)
{
    ScaledFunctionAB<ChebFunction> cf(a, b,
        ChebFunction(ScaledFunctionM11<FUNCTION>(a, b, f), 32));
    pair<double, double> errors = testInterpolant(f, cf, a, b);
    assert(errors.first < highPrecEps);
}
void testChebAuto()
{
    testChebAutoHelper(EPX(), 0, 1);
}

template<typename FUNCTION> void testSplineAutoHelper(FUNCTION const& f, double a, double b)
{
    int nIntervals = 36;//nIntervals + 1 POINTS
    Vector<pair<double, double> > xy = generateEqualSpacedPoints(a, b, nIntervals, f);
    NotAKnotCubicSplineInterpolation no(xy);
    pair<double, double> errors = testInterpolant(f, no, a, b);
    assert(errors.first < highPrecEps);
}
void testSplineAuto()
{
    testSplineAutoHelper(CUBE(), 0, 1);
}*/

void testInterpolGivenPoints()
{

}
struct SQUARE2
{
    double operator()(double x)const{return x * x - 0.25;}
};
void testChebRoots()
{
    ChebFunction cf(SQUARE2(), 16);
    Vector<double> rroots = cf.findAllRealRoots();
    DEBUG("roots");
    rroots.debug();//expect 0.5 and -0.5
    rroots = findAllRealRootsCheb(SQUARE2(), -1, 1);
    DEBUG("roots adaptive");
    rroots.debug();//expect 0.5 and -0.5
    rroots = findAllRealRootsCheb(TestFunctions1D::Sin(), -10, 10);
    DEBUG("sin roots adaptive");
    rroots.debug();//expect 0 and +- iPi for i 1 to 3
}


//from Fausett; ignore x
struct FPB1 : public MultivarFuncHelper::F1DBase
{
    double operator()(Vector<double> const& yx)const{++evalCount2; return yx[1];}
};
struct FPB2 : public MultivarFuncHelper::F1DBase
{
    double operator()(Vector<double> const& yx)const{return 2 * yx[0] * yx[1];}
};
struct FPBBV
{
    double evaluateGoal(Vector<double> const& yGoal)const{return yGoal[0] + yGoal[1] - 0.25;}
    Vector<double> getInitial(double b)const
    {
        Vector<double> y0(2);
        y0[0] = 1;
        y0[1] = b;
        return y0;
    };
};
void testBoundaryValue()
{
    MultivarFuncHelper f;
    FPB1 f1;
    FPB2 f2;
    FPStiffDummy fd;
    f.fs.append(&f1);
    f.fs.append(&f2);
    f.fs.append(&fd);
    Vector<double> xPoints(5);
    for(int i = 1; i <= 5; ++i) xPoints[i - 1] = i * 0.2;
    Vector<Vector<double> > result = solveBoundaryValue(f, 0, 1, xPoints, FPBBV());
    DEBUG("result");
    for(int i = 0; i < result.getSize(); ++i)
    {
        DEBUG(i);
        DEBUG(xPoints[i]);
        DEBUG("result[i]");
        result[i].debug();
    }
    DEBUG(evalCount2);
    evalCount2 = 0;
}

struct MultivarITest1
{//from https://en.wikipedia.org/wiki/Multiple_integral
    //for more see https://www.wolframalpha.com/examples/Integrals.html
    int getD()const{return 2;}
    double operator()(Vector<double> const& x)const
    {
        assert(x.getSize() == getD());
        return x[0] * x[0] + 4 * x[1];
    }
    Vector<pair<double, double> > getBox()const
    {
        Vector<pair<double, double> > box(2);
        box[0] = make_pair(11.0, 14.0);
        box[1] = make_pair(7.0, 10.0);
        return box;
    }
    double solution()const{return 1719;}
};
template<typename INTEGRATOR1D, typename FUNCTION>
testMultidimIntegrationHelper(int evalsPerDim = 33)
{
    FUNCTION f;
    DEBUG(f.solution());
    RecursiveIntegralFunction<FUNCTION, INTEGRATOR1D> rif(f.getBox(),
        evalsPerDim);
    pair<double, double> result = rif.integrate();
    DEBUG(result.first);
    DEBUG(result.second);
}
void testMultidimIntegration()
{//also try smooth functions like multidim exponential on 01 or multidim normal pdf because know answer! good choice
    int nEvals = 1000000;
    testMultidimIntegrationHelper<Cheb1DIntegrator, MultivarITest1>(
        nextPowerOfTwo(pow(nEvals, 1.0/MultivarITest1().getD())/2) + 1);//for Cheb specifically
    MultivarITest1 f;
    double resultMonte = MonteCarloIntegrate(f.getBox(), nEvals, InsideTrue(), f).first;
    DEBUG(resultMonte);
    double resultSobol = SobolIntegrate(f.getBox(), nEvals, InsideTrue(), f).first;
    DEBUG(resultSobol);
}

void FFTTestReal()
{
    int n = 4;
    Vector<double> x(n);
    for(int i = 0; i < n; ++i) x[i] = i;
    Vector<complex<double> > z(n);
    for(int i = 0; i < n; ++i) z[i] = complex<double>(x[i], 0);
    double normYDiff = normInf(FFTRealEven(x) - FFTGeneral(z));
    DEBUG(normYDiff);
    DEBUG("FFTRealEven(x)");
    FFTRealEven(x).debug();
    DEBUG("FFTGeneral(z))");
    FFTGeneral(z).debug();
}

Vector<double> slowDCTI(Vector<double> const& x)
{
    int n = x.getSize() - 1;
    Vector<double> result(n + 1);
    for(int i = 0; i <= n; ++i)
    {
        double ci = 0;
        for(int j = 0; j <= n; ++j) ci += cos(i * j * PI()/n)
            * x[j] * (j == 0 || j == n ? 0.5 : 1.0);
        result[i] = ci;
    }
    return result;
}
void DCTTestHelper(Vector<double> const& x, double eps = defaultPrecEps)
{
    double normXDiff = normInf(x - IDCTI(DCTI(x))),
        normYDiff = normInf(slowDCTI(x) - DCTI(x));
    if(normXDiff >= eps || normYDiff >= eps)
    {
        DEBUG("failed for x=");
        DEBUG(x.getSize());
        x.debug();
        DEBUG(normXDiff);
        DEBUG(normYDiff);
        DEBUG("IDCTI(DCTI(x))");
        IDCTI(DCTI(x)).debug();
        DEBUG("DCTI(x)");
        DCTI(x).debug();
        DEBUG("slowDCTI(x)");
        slowDCTI(x).debug();
        assert(false);
    }
}
void DCTTestAuto()
{
    int nMax = 100, nn = 1000;
    for(int n = 3; n <= nMax; ++n)//fails for 2 in bits
    {
        for(int j = 0; j < nn; ++j)
        {
            Vector<double> x(n);
            for(int i = 0; i < n; ++i) x[i] = GlobalRNG().uniform(-1, 1);
            DCTTestHelper(x);
        }
    }
    DEBUG("DCTTestAuto passed");
}

void testAllAuto()
{
    testELessAuto();
    FFTTestAuto();
    DCTTestAuto();
}

struct SolveTestFunctions1D
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
            return (*f)(x);
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
int SolveTestFunctions1D::evalCount = 0;


void debugEq1DResult(pair<double, double> const& result, double answer,
    Vector<Vector<string> >& matrix)
{
    DEBUG(normInf(result.first - answer));
    DEBUG(result.second);
    DEBUG(SolveTestFunctions1D::evalCount);
    matrix.lastItem().append(toString(normInf(result.first - answer)));
    matrix.lastItem().append(toString(result.second));
    matrix.lastItem().append(toString(SolveTestFunctions1D::evalCount));
    SolveTestFunctions1D::evalCount = 0;
}
template<typename FUNCTION> void testNonlinearEqHelper1D(FUNCTION const& f,
    double x0, Vector<Vector<string> >& matrix)
{
    DEBUG("ExpSearch");
    matrix.lastItem().append("ExpSearch");
    debugEq1DResult(exponentialSearch(f, x0), f.getAnswer(), matrix);
    DEBUG("Secant");
    matrix.lastItem().append("Secant");
    debugEq1DResult(solveSecant(f, x0), f.getAnswer(), matrix);
    DEBUG("SecantGlobal");
    matrix.lastItem().append("SecantGlobal");
    debugEq1DResult(solveSecantGlobal(f), f.getAnswer(), matrix);
}

void testNonlinearEq1D()
{
    Vector<Vector<string> > matrix;
    Vector<SolveTestFunctions1D::MetaF> fs = SolveTestFunctions1D::getFunctions();
    for(int i = 0; i < fs.getSize(); ++i)
    {
        string name = fs[i].getName();
        DEBUG(name);
        matrix.append(Vector<string>());
        matrix.lastItem().append(name);
        testNonlinearEqHelper1D(fs[i], fs[i].getX0(), matrix);
    }
    int reportNumber = time(0);
    string filename = "reportNonlinearEq1D" + toString(reportNumber) + ".csv";
    createCSV(matrix, filename.c_str());
}

struct GradTest
{
    double operator()(Vector<double> const& x)const
    {
        return x[0] * x[0] + x[1];
    }
    Vector<double> grad(Vector<double> const& x)const
    {
        Vector<double> result(2);
        result[0] = 2 * x[0];
        result[1] = 1;
        return result;
    }
    double dd(Vector<double> const& x, Vector<double> const& d)
    {
        return dotProduct(grad(x), d);
    }
};
void testGradDD()
{
    GradTest g;
    Vector<double> x(2, 5), d(2, 1);
    DEBUG(normInf(g.grad(x) - estimateGradientCD(x, g)));
    DEBUG(abs(g.dd(x, d) - estimateDirectionalDerivativeCD(x, g, d)));
}

int main()
{
    testNonlinearEq();
    return 0;
    testNonlinearEq1D();//fails here check again
    return 0;
    testODE();
    return 0;
    testIntegrators();
    return 0;
    testInterpolChosenPoints();
    return 0;
    testDeriv();
    return 0;
    testAllAutoNumericalMethods();
    return 0;
    FFTTestReal();
    return 0;
    testGradDD();
    return 0;
    testMultidimIntegration();
    return 0;
    testELessAuto();
    return 0;
    return 0;
    testChebRoots();
    return 0;
    testStiffSolvers();
    return 0;
    testBoundaryValue();
    return 0;
    //testSplineAuto();
    return 0;
    //testChebAuto();
    return 0;
    testAllAuto();
    return 0;
    testAllRootSolver();
    return 0;
    testRungKutta();
    return 0;

    DEBUG(numeric_limits<double>::min());
    DEBUG(numeric_limits<double>::max());
    DEBUG(numeric_limits<double>::epsilon());

    return 0;
}
