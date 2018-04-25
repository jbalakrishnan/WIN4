"""
Chabauty--Coleman method for hyperelliptic curves of genus 3, with Jacobian of rank 1.

------------
The main function is chabauty_test(C,p,P,L,prec,n)

INPUT:
- C: a hyperelliptic curve of genus 3 over Q whose Jacobian J has rank 1;
      C should be given by a model y^2=f(x) where f\in Q[x] has degree 7.
- p: an odd prime of good reduction for C not dividing the leading coeff of f;
      we recommend p >= 7.
- P: a point in C(Q) such that [P-infty]\in J(Q) has infinite order.
- L: a list of known rational points on C.
- prec: working p-adic precision (recommended: 2*p+4)
- n: working t-adic precision (recommended: 2*p+4)

OUTPUT:
- A list of all points in C(Q)\setminus L up to hyperelliptic involution.
  Note that if R is in L but (R[0],-R[1]) is not in L, this point will NOT be returned.
- A list of all points in C(Q_p)\setminus C(Q) with y-coordinate = 0
- A list of all points R\in C(Q_p) such that [R-infty]\in J(Q_p)_{tors}\setminus J(Q_p)[2],
  up to hyperelliptic involution. If R comes from a point over K with [K:Q]=2, the coordinates in K
  are given as the corresponding minimal polynomials over Q.
- A list of all remaining points R in C(Q_p) such that [R-infty] is in the p-adic closure of J(Q) in J(Q_p).
  If R is defined over K with [K:Q] = 2, the coordinates are given as minimal polynomials.


AUTHORS:
- Jennifer Balakrishnan, Francesca Bianchi, Mirela Ciperiani, Victoria Cantoral-Farfan, Anastassia Etropolski (2018)


EXAMPLES:

Example 1: new rational points
sage: R.<x> = PolynomialRing(Rationals())
sage: f = -4*x^7 - 11*x^6 + 22*x^5 + x^4 - 18*x^3 - 2*x^2 + 4*x + 1
sage: C = HyperellipticCurve(f)
sage: p = 7
sage: prec = 2*p+4
sage: P = C(0,-1)
sage: a,b,c,d = chabauty_test(C,p,P,[],prec,prec)
sage: a
[(-18/49 : -20689/823543 : 1), (0 : 1 : 1), (0 : 1 : 0)]


Example 2: Example 3.3 (Section 4.1)
sage: R.<x> = PolynomialRing(Rationals())
sage: f = 4*x^7 - 15*x^6 + 32*x^5 - 38*x^4 + 32*x^3 - 15*x^2 + 4*x
sage: C = HyperellipticCurve(f)
sage: P = C(1,-2)
sage: L = [C(0,1,0),P,C(0,0)]
sage: p = 11
sage: prec = 2*p + 4
sage: a,b,c,d = chabauty_test(C,p,P,L,prec,prec)
sage: len(a)
0
sage: len(b)
2
sage: len(c)
0
sage: d
[(-1, x^2 + 140)]
"""

############## AUXILIARY FUNCTIONS ##############

def ann_basis(C,P,p,prec):
    """
    Constructs the pullback {alpha, beta} on C
    of a basis for Ann(J(Q)). Here C is embedded
    into J via infty.
    """
    K = pAdicField(p,prec)
    CK = C.change_ring(K)
    w = CK.invariant_differential()
    x, y = CK.monsky_washnitzer_gens()
    T = CK([0,1,0])
    diffs = [w,x*w,x^2*w]
    if P[1]%p != 0:
        ints = [CK.coleman_integral(w,T,CK(P)), CK.coleman_integral(x*w,T,CK(P)), CK.coleman_integral(x^2*w,T,CK(P))]
    else:
        Cp = C.change_ring(GF(p))
        Q = Cp(P)
        W = Q_lift(CK,Q,p)
        ints = list(CK.tiny_integrals([1,x,x^2],W,CK(P)))
    if ints[0] == 0:
        alpha = vector([1,0,0])
        m = min([valuation(ints[1]),valuation(ints[2])])
        beta = vector([0,-ints[2],ints[1]])/p^m
    else:
        m1 = min([valuation(ints[0]),valuation(ints[1])])
        alpha = vector([-ints[1],ints[0],0])/p^m1
        m2 = min([valuation(ints[0]),valuation(ints[2])])
        beta = vector([-ints[2],0,ints[0]])/p^m2
        m = max(m1,m2)
    C._prec_lost_annbasis = m
    C._new_prec = min(x.precision_absolute() for x in [alpha[0], alpha[1], alpha[2], beta[0], beta[1], beta[2]])
    assert C._new_prec >= 6, "Input p-adic precision too low"
    return [[alpha,alpha[0]*diffs[0] + alpha[1]*diffs[1] + alpha[2]*diffs[2]],[beta, beta[0]*diffs[0] + beta[1]*diffs[1] + beta[2]*diffs[2]]]


def order_of_vanishing(Q,alphacoeffs,betacoeffs,Cp,p):
    """
    Given a point Q in C(Fp) and alpha and beta (given as alphacoeffs and betacoeffs),
    calculates the order of vanishing of alpha and beta at Q.
    """
    xQ,yQ = Cp.local_coord(Q)
    dx = xQ.derivative()
    alphacoeffsp = alphacoeffs.change_ring(GF(p))
    betacoeffsp = betacoeffs.change_ring(GF(p))
    alphabar = (alphacoeffsp[0]+alphacoeffsp[1]*xQ+alphacoeffsp[2]*xQ^2)/(2*yQ)*dx
    betabar = (betacoeffsp[0]+betacoeffsp[1]*xQ+betacoeffsp[2]*xQ^2)/(2*yQ)*dx
    return [alphabar.valuation(),betabar.valuation()]


def upper_bound(Q,alphacoeffs,betacoeffs,Cp,p):
    """
    Given Q in Cp(Fp) finds an upper bound for the number
    of rational points in the residue disk of Q by calculating the
    order of vanishing of alpha and beta (given as alphacoeffs and betacoeffs).
    """
    [m1,m2] = order_of_vanishing(Q,alphacoeffs,betacoeffs,Cp,p)
    m = min([m1,m2])
    if m < p-2:
        return m+1
    else:
        return "We cannot extract information from this prime on this residue disk"


def Q_lift(CK,Q,p):
    """
    For a residue disk Q, computes the Teichmueller point lifting Q to the curve CK,
    where CK should be over the p-adics.
    """
    xQ = Integers()(Q[0])
    yQ = Integers()(Q[1])
    if yQ == 0:
        r = CK.hyperelliptic_polynomials()[0].roots()
        Q_lift = CK(exists(r, lambda a : (Integers()(a[0]) - xQ) % p == 0)[1][0],0)
    else:
        K = CK.base_ring()
        xQ = K.teichmuller(K(xQ))
        lifts = CK.lift_x(xQ,all=True)
        for i in range(len(lifts)):
            if (Integers()(lifts[i][1])-yQ) % p == 0:
                Q_lift = lifts[i]
    return Q_lift


def normalize_and_truncate(C,f,n,p):
    """
    Takes a power series f and normalizes it so that 0 is not a root and so that f(pt) is nonzero mod p,
    then truncates it to a polynomial of sufficiently high precision.
    """
    K = f[0].parent()
    fp = f(K(p)*f.parent().gen())
    k = min(x.valuation() for x in fp)
    C._prec_lost_powseries = k
    upperbd = max([i for i in range(fp.degree() + 1) if fp[i].valuation() == k])
    assert upperbd <= C._prec_lost_powseries, "Minimum valuation achieved later than it should be."
    fp = fp/p^k
    n = C._new_prec
    M = n
    r = n
    while True:
        if r % p == 0:
            break
        else:
            r = r + 1
    if r - r.valuation(p) < n:
        M = r + 1
    assert M <= fp.precision_absolute, "Input power series precision is too low."
    return fp.truncate(M)


def power_series_zeros(C,f,n,p):
    """
    Assumes discriminant of the power series f is nonzero and fp has been normalized.
    """
    R.<t> = PolynomialRing(f[0].parent())
    upperbd = max([i for i in range(f.degree() + 1) if f[i].valuation() == 0])
    if f.newton_polygon().slopes()[0] > 0:
        return []
    else:
        roots = list(gp.polrootspadic(f,p,1))
    minprec = min(x.precision_absolute() for x in f)
    rootprec = C._new_prec - C._prec_lost_powseries
    rootprec = min(rootprec,minprec)
    roots = [p*x + O(p^rootprec) for x in roots if x.valuation(p)>= 0]
    assert upperbd >= len(roots), "More roots found than should exist by Strassman! Precision issue?"
    return roots



############## MAIN FUNCTION ###############

def chabauty_test(C,p,P,L,prec,n):

    #Handle projective coordinates
    L0 = list(L)
    L = []
    for a in L0:
        if a[2] != 1 and a[2] != 0:
            L.append(C(a[0]/a[2],a[1]/(a[2]^4),1))
        else:
            L.append(a)
    L = list(L)
    if P[2]!=1 and P[2]!=0:
        P = C(P[0]/P[2],P[1]/(P[2])^4,1)

    #Handle non-monic polynomial case
    def_poly = C.hyperelliptic_polynomials()[0]
    lead_coeff = def_poly.leading_coefficient()
    assert lead_coeff % p != 0, "The leading coefficient is divisible by p."
    def_poly_new = def_poly(lead_coeff*def_poly.parent().0)/lead_coeff^8
    C_old = C
    C = HyperellipticCurve(def_poly_new)
    L_old = L
    P_old = P
    if P_old[2] == 0:
        P = C(0,1,0)
    else:
        P = C(P_old[0]/lead_coeff,P_old[1]/lead_coeff^4)
    L = []
    for a in L_old:
        if a[2] == 0:
            L.append(C(0,1,0))
        else:
            L.append(C(a[0]/lead_coeff,a[1]/lead_coeff^4))
    L = list(set(L))

    #Handle hyperelliptic involution conjugates
    for a in L:
        if a == C(0,1,0):
            continue
        minusa = C(a[0],-a[1])
        if minusa not in L:
            L.append(minusa)

    if C(0,1,0) not in L:
        L.append(C(0,1,0))

    C._prec = prec
    K = pAdicField(p,prec)
    Cp = C.change_ring(GF(p))
    CK = C.change_ring(K)

    Basis = ann_basis(C,P,p,prec)
    alpha = Basis[0][1]
    beta = Basis[1][1]
    alphacoeffs = Basis[0][0]
    betacoeffs = Basis[1][0]
    RatClasses = []
    RatClassesModp = []
    for Q in L:
        if Q[0].valuation(p) < 0:
            Qbar = Cp(0,1,0)
        else:
            Qbar = Cp(Q)
        if ZZ(Qbar[1])>p/2:
            continue
        if Qbar not in RatClassesModp:
            RatClassesModp.append(Qbar)
            RatClasses.append([Q])
        else:
            i = 0
            while RatClassesModp[i] != Qbar:
                i = i+1
            RatClasses[i].append(Q)

    RatClassesWithPoints0 = [[RatClassesModp[i],RatClasses[i]] for i in range(len(RatClasses))]
    ClassesToConsider = [Q for Q in list(Cp.points()) if ZZ(Q[1])<p/2]
    RatClassesWithPoints = list(RatClassesWithPoints0)

    for i in range(len(RatClassesModp)):
        if len(RatClasses[i]) == upper_bound(RatClassesModp[i],alphacoeffs,betacoeffs,Cp,p):
            ClassesToConsider.remove(RatClassesModp[i])
        else:
            RatClassesWithPoints.remove(RatClassesWithPoints0[i])

    ClassesToConsiderQ = [pt for pt in ClassesToConsider if pt in RatClassesModp]
    ClassesToConsiderNotQ = [pt for pt in ClassesToConsider if pt not in RatClassesModp]
    ClassesNotToConsider = [pt for pt in RatClassesModp if pt not in ClassesToConsider]

    AllPowerSeries = []
    AllParametrisations = []
    for Q in ClassesToConsider:
        if Q in ClassesToConsiderQ and Q[1]!=0:
            i = RatClassesModp.index(Q)
            R = RatClasses[i][0]
            xR,yR = CK.local_coord(R,prec=n+2)
            AllParametrisations.append([xR,yR])
            dx = xR.derivative()
            alphaR = (alphacoeffs[0]+alphacoeffs[1]*xR+alphacoeffs[2]*xR^2)/(2*yR)*dx
            betaR = (betacoeffs[0]+betacoeffs[1]*xR+betacoeffs[2]*xR^2)/(2*yR)*dx
            alphaRint = alphaR.integral()
            betaRint = betaR.integral()
        else:
            R = Q_lift(CK,Q,p)
            xR,yR =  CK.local_coord(R,prec=n+2)
            AllParametrisations.append([xR,yR])
            dx = xR.derivative()
            alphaR = (alphacoeffs[0]+alphacoeffs[1]*xR+alphacoeffs[2]*xR^2)/(2*yR)*dx
            betaR = (betacoeffs[0]+betacoeffs[1]*xR+betacoeffs[2]*xR^2)/(2*yR)*dx
            alphaRint = alphaR.integral()
            betaRint = betaR.integral()
            alphaRint = alphaRint + CK.coleman_integral(alpha,CK(0,1,0), R)
            betaRint = betaRint + CK.coleman_integral(beta,CK(0,1,0),R)
        Series = []
        for f in [alphaRint,betaRint]:
            T.<t> = PowerSeriesRing(f[0].parent())
            if f.parent() == T:
                f = T(f)
            else:
                f = f.power_series()
            Series.append(f)
        AllPowerSeries.append([Q,Series])
    NewPoints = []

    for i in range(len(AllPowerSeries)):
        h = AllPowerSeries[i]
        [f,g] = h[1]
        if h[0] in ClassesToConsiderQ and h[0][1]!=0:
            j = RatClassesModp.index(h[0])
            tval_f = f.valuation()
            tval_g = g.valuation()
            f = f/((f.parent().gen())^tval_f)
            g = g/((g.parent().gen())^tval_g)
            assert f.truncate().discriminant() != 0 or g.truncate().discriminant() != 0, "Both power series have repeated roots."
            if f.truncate().discriminant() != 0:
                f_p = normalize_and_truncate(C,f,n,p)
                Zerosf = [x for x in power_series_zeros(C,f_p,n,p)]
                CommonZeros = [x for x in Zerosf if g.truncate()(x) == 0]
            else:
                g_p = normalize_and_truncate(C,g,n,p)
                Zerosg = [x for x in power_series_zeros(C,g_p,n,p)]
                CommonZeros = [x for x in Zerosg if f.truncate()(x) == 0]
            xh = AllParametrisations[i][0]
            yh = AllParametrisations[i][1]
            NewPoints.extend([CK(xh(K(sage_eval('%s'%t0))),yh(K(sage_eval('%s'%t0)))) for t0 in CommonZeros])
        else:
            tval_f = f.valuation()
            tval_g = g.valuation()
            f = f/((f.parent().gen())^tval_f)
            g = g/((g.parent().gen())^tval_g)
            assert f.truncate().discriminant() != 0 or g.truncate().discriminant() != 0, "Both power series have repeated roots."
            if f.truncate().discriminant() != 0:
                f_p = normalize_and_truncate(C,f,n,p)
                Zerosf = [x for x in power_series_zeros(C,f_p,n,p)]
                CommonZeros = [x for x in Zerosf if g.truncate()(x) == 0]
            else:
                g_p = normalize_and_truncate(C,g,n,p)
                Zerosg = [x for x in power_series_zeros(C,g_p,n,p)]
                CommonZeros = [x for x in Zerosg if f.truncate()(x) == 0]
            if tval_f > 0 and tval_g > 0:
                CommonZeros.append(0)
            xh = AllParametrisations[i][0]
            yh = AllParametrisations[i][1]

            NewPoints.extend([CK(xh(K(sage_eval('%s'%t0))),yh(K(sage_eval('%s'%t0)))) for t0 in CommonZeros])

    lnew = len(NewPoints)
    NewRatPoints = []
    TwoTorsPoints = []
    OtherTorsPoints = []
    QuestionPoints=[]
    for A in NewPoints:
        try:
            NRP = C.lift_x(QQ(A[0]))
            if NRP[1] - A[1] == 0:
                NewRatPoints.append(NRP)
            else:
                NRP = C(NRP[0],-NRP[1])
                NewRatPoints.append(NRP)
        except ValueError:
            T = CK([0,1,0])
            if CK.is_in_weierstrass_disc(A):
                if A[1] == 0:
                    I0 = 0
                    I1 = 0
                    I2 = 0
                    TwoTorsPoints.append(A)
                else:
                    W = CK.find_char_zero_weier_point(A)
                    I0,I1,I2, _,_,_ = CK.tiny_integrals_on_basis(W, A)
            else:
                I0,I1,I2, _,_,_ = CK.coleman_integrals_on_basis(T,A)
                if I0 == 0 and I1 == 0 and I2 == 0:
                    OtherTorsPoints.append(A)
                else:
                    QuestionPoints.append(A)

    for A in L:
        if A in NewRatPoints:
            NewRatPoints.remove(A)

    for A in NewRatPoints:
        if A[0].valuation(p)<0:
            for B in NewRatPoints:
                if B[0].valuation(p) >= 0 or B == A:
                    continue
                if B[0] == A[0] and B[1] == -A[1]:
                    NewRatPoints.remove(B)

    #Back to non-monic equation
    if def_poly.is_monic() == False:
        C_old_K = C_old.change_ring(K)
        NewRatPoints_new = NewRatPoints
        TwoTorsPoints_new = TwoTorsPoints
        OtherTorsPoints_new = OtherTorsPoints
        QuestionPoints_new = QuestionPoints
        NewRatPoints = [C_old(a[0]*lead_coeff,a[1]*lead_coeff^4,a[2]) for a in NewRatPoints_new]
        TwoTorsPoints = [C_old_K(a[0]*lead_coeff,a[1]*lead_coeff^4,a[2]) for a in TwoTorsPoints_new]
        OtherTorsPoints = [C_old_K(a[0]*lead_coeff,a[1]*lead_coeff^4,a[2]) for a in OtherTorsPoints_new]
        QuestionPoints = [C_old_K(a[0]*lead_coeff,a[1]*lead_coeff^4,a[2]) for a in QuestionPoints_new]

    if C_old(0,1,0) not in L_old and (1,0,0) not in L_old:
        NewRatPoints.append(C_old(0,1,0))


    for A in OtherTorsPoints:
        i = OtherTorsPoints.index(A)
        p2 = algdep(A[1],2)
        p1 = algdep(A[0],2)
        Lf.<par> = NumberField(p2)
        if p1.degree() == 1:
            try:
                OTP = C_old.change_ring(Lf).lift_x(QQ(A[0]))
                if p2(OTP[1]) == 0 or p2(-OTP[1]) ==0:
                    OtherTorsPoints[i] = (QQ(A[0]),p2)
            except ValueError:
                pass
        else:
            Lf.<par> = NumberField(p1)
            try:
                OTP = C_old.change_ring(Lf).lift_x(par)
                if p2(OTP[1]) == 0 or p2(-OTP[1]) ==0:
                    if p2.degree() == 1:
                        OtherTorsPoints[i] = (p1,QQ(A[2]))
                    else:
                        OtherTorsPoints[i] = (p1,p2)
                else:
                    continue
            except ValueError:
                continue

    for A in QuestionPoints:
        i = QuestionPoints.index(A)
        p2 = algdep(A[1],2)
        p1 = algdep(A[0],2)
        Lf.<par> = NumberField(p2)
        if p1.degree() == 1:
            try:
                OTP = C_old.change_ring(Lf).lift_x(QQ(A[0]))
                if p2(OTP[1]) == 0 or p2(-OTP[1]) ==0:
                    QuestionPoints[i] = (QQ(A[0]),p2)
            except ValueError:
                pass
        else:
            Lf.<par> = NumberField(p1)
            try:
                OTP = C_old.change_ring(Lf).lift_x(par)
                if p2(OTP[1]) == 0 or p2(-OTP[1]) == 0:
                    if p2.degree() == 1:
                        QuestionPoints[i] = (p1,QQ(A[2]))
                    else:
                        QuestionPoints[i] = (p1,p2)
                else:
                    continue
            except ValueError:
                continue

    OtherTorsPoints = list(set(OtherTorsPoints))
    QuestionPoints = list(set(QuestionPoints))

    return NewRatPoints, TwoTorsPoints, OtherTorsPoints, QuestionPoints