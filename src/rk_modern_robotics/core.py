import sympy as sp
import numpy as np
a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z = sp.symbols('a b c d e f g h i j k l m n o p q r s t u v w x y z')

I3=sp.eye(3)
I4=sp.eye(4)

def RotInv(R):
    return np.transpose(R)

def VecToso3(omega):
    return np.array([[0, -omega[2], omega[1]],
                      [omega[2], 0, -omega[0]],
                      [-omega[1], omega[0], 0]])

def so3ToVec(so3mat):
    return sp.Matrix([so3mat[2,1], so3mat[0,2], so3mat[1,0]])

def AxisAng3(expc3):
    th=sp.Matrix(expc3)
    theta=th.norm()
    return (list(th/theta), theta)

def MatrixExp3(so3mat):
    theta = AxisAng3(so3ToVec(so3mat))[1]
    omega_hat = sp.Matrix(so3mat)/theta
    return (I3+omega_hat*sp.sin(theta)+omega_hat**2*(1-sp.cos(theta))).evalf()

def NearZero(z):
    # Checks if a number is near zero; returns False if z contains free symbols
    z_sym = sp.sympify(z)
    if z_sym.free_symbols:
        return z_sym == 0
    return abs(float(z_sym)) < 1e-6

def MatrixLog3(R):
    R = sp.Matrix(R)
    acosinput = sp.simplify((R.trace() - 1) / sp.Integer(2))

    # Case 1: Symbolic input (comparisons like >= 1 cannot be evaluated directly)
    if acosinput.free_symbols:
        theta = sp.acos(acosinput)
        return (theta / (2 * sp.sin(theta)) * (R - R.T)).evalf()

    # Case 2: Numeric / known values
    if acosinput >= 1:
        return sp.zeros(3, 3)
    elif acosinput <= -1:
        if not NearZero(1 + R[2, 2]):
            omg = (1 / sp.sqrt(2 * (1 + R[2, 2]))) * sp.Matrix([R[0, 2], R[1, 2], 1 + R[2, 2]])
        elif not NearZero(1 + R[1, 1]):
            omg = (1 / sp.sqrt(2 * (1 + R[1, 1]))) * sp.Matrix([R[0, 1], 1 + R[1, 1], R[2, 1]])
        else:
            omg = (1 / sp.sqrt(2 * (1 + R[0, 0]))) * sp.Matrix([1 + R[0, 0], R[1, 0], R[2, 0]])
        return (VecToso3(sp.pi * omg)).evalf()
    else:
        theta = sp.acos(acosinput)
        return (theta / (2 * sp.sin(theta)) * (R - R.T)).evalf()

def RpToTrans(R, p):
    return sp.Matrix([[sp.Matrix(R),sp.Matrix(p)],[0,0,0,1]])

class TupleOut(tuple):
    def __str__(self):
        return "\n".join(str(item) for item in self)

    def __repr__(self):
        return self.__str__()
    
def TransToRp(T):
    T=sp.Matrix(T)
    return TupleOut((T[0:3,0:3],T[0:3,3]))

def TransInv(T):
    T=sp.Matrix(T)
    R=T[0:3,0:3]
    p=T[0:3,3]
    Rt=R.T
    return sp.Matrix([[Rt,-Rt*p],[0,0,0,1]])

def VecTose3(V):
    V=sp.Matrix(V)
    return sp.Matrix([[0,-V[2],V[1],V[3]],[V[2],0,-V[0],V[4]],[-V[1],V[0],0,V[5]],[0,0,0,0]])

def se3ToVec(se3mat):
    se3mat=sp.Matrix(se3mat)
    return sp.Matrix([se3mat[2,1],se3mat[0,2],se3mat[1,0],se3mat[0,3],se3mat[1,3],se3mat[2,3]])

def Adjoint(T):
    T=sp.Matrix(T)
    R=T[0:3,0:3]
    posi=T[0:3,3]
    return sp.Matrix([[R,sp.zeros(3,3)],[VecToso3(posi)*R,R]])

def _format_out(expr):
    expr = sp.simplify(expr)
    return expr if expr.free_symbols else expr.evalf()

def ScrewToAxis(q, s, h):
    q = sp.Matrix(q)
    s = sp.Matrix(s)
    v = q.cross(s) + h * s
    return (s.col_join(v)).evalf()

def AxisAng6(expc6):
    th=sp.Matrix(expc6)
    theta=sp.Matrix(th[0:3]).norm()
    return ((th/theta).evalf(), theta.evalf())

def MatrixExp6(se3mat):
    se3mat = sp.Matrix(se3mat)
    omgtheta = so3ToVec(se3mat[0: 3, 0: 3])
    if omgtheta == sp.zeros(3, 1) or (not omgtheta.free_symbols and NearZero(omgtheta.norm())):
        return sp.Matrix([[sp.eye(3), se3mat[0: 3, 3]], [0, 0, 0, 1]]).evalf()
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        R = MatrixExp3(se3mat[0: 3, 0: 3])
        p = (sp.eye(3) * theta + (1 - sp.cos(theta)) * omgmat + (theta - sp.sin(theta)) * omgmat**2) * se3mat[0: 3, 3] / theta
        return sp.Matrix([[R, p.evalf()], [0, 0, 0, 1]])

def MatrixLog6(T):
    T = sp.Matrix(T)
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if omgmat == sp.zeros(3, 3):
        return sp.Matrix([[sp.zeros(3, 3), p], [0, 0, 0, 0]])
    else:
        theta = AxisAng3(so3ToVec(omgmat))[1]
        G_inv = (sp.eye(3) 
                 - omgmat / 2 
                 + (1 / theta - sp.Integer(1) / (2 * sp.tan(theta / 2))) * (omgmat ** 2) / theta)
        v_theta = G_inv * p
        return _format_out(sp.Matrix([[omgmat, v_theta], [0, 0, 0, 0]]))

def ProjectToSO3(mat):
    mat = sp.Matrix(mat)
    if not mat.free_symbols:
        mat_np = np.array(mat.tolist(), dtype=float)
        U, s, Vh = np.linalg.svd(mat_np)
        R = np.dot(U, Vh)
        if np.linalg.det(R) < 0:
            # In this case the result may be far from mat.
            R[:, 2] = -R[:, 2]
        return sp.Matrix(R)
    else:
        # Symbolic Gram-Schmidt orthonormalization onto SO(3)
        c0 = mat[:, 0]
        c1 = mat[:, 1]

        r0 = c0 / sp.sqrt(c0.dot(c0))
        v1 = c1 - (r0.dot(c1)) * r0
        r1 = v1 / sp.sqrt(v1.dot(v1))
        r2 = r0.cross(r1)  # Cross product guarantees orthogonality and det(R) = +1

        R = r0.row_join(r1).row_join(r2)
        return sp.simplify(R)

def ProjectToSE3(mat):
    mat = sp.Matrix(mat)
    return RpToTrans(ProjectToSO3(mat[:3, :3]), mat[:3, 3])

def DistanceToSO3(mat):
    mat = sp.Matrix(mat)
    det = sp.simplify(mat.det())
    try:
        if det <= 0:
            return 1e+9
    except TypeError:
        # det contains free symbols whose sign is unknown (e.g., det = a)
        pass
    diff = mat.T * mat - sp.eye(3)
    dist = sp.sqrt(sp.trace(diff.T * diff))
    return _format_out(dist)
def DistanceToSE3(mat):
    mat = sp.Matrix(mat)
    matR = mat[0:3, 0:3]
    det = sp.simplify(matR.det())
    try:
        if det <= 0:
            return 1e+9
    except TypeError:
        # det contains free symbols whose sign cannot be determined
        pass
    M = (matR.T * matR).row_join(sp.zeros(3, 1)).col_join(mat[3, :])
    diff = M - sp.eye(4)
    dist = sp.sqrt(sp.trace(diff.T * diff))
    return _format_out(dist)

def TestIfSO3(mat):
    return abs(DistanceToSO3(mat)) < 1e-3


def TestIfSE3(mat):
    return abs(DistanceToSE3(mat)) < 1e-3

def FKinBody(M, Blist, thetalist):
    T = sp.Matrix(M)
    Blist = sp.Matrix(Blist)
    thetalist = sp.Matrix(thetalist)
    for i in range(len(thetalist)):
        T = T * MatrixExp6(VecTose3(Blist[:, i] * thetalist[i]))
    return _format_out(sp.trigsimp(T))

def FKinSpace(M, Slist, thetalist):
    T = sp.Matrix(M)
    Slist = sp.Matrix(Slist)
    thetalist = sp.Matrix(thetalist)
    for i in range(len(thetalist) - 1, -1, -1):
        T = MatrixExp6(VecTose3(Slist[:, i] * thetalist[i])) * T
    return _format_out(sp.trigsimp(T))

def JacobianBody(Blist, thetalist):
    Blist = sp.Matrix(Blist)
    thetalist = sp.Matrix(thetalist)
    Jb = sp.Matrix(Blist)
    T = sp.eye(4)
    for i in range(len(thetalist) - 2, -1, -1):
        T = T * MatrixExp6(VecTose3(Blist[:, i + 1] * -thetalist[i + 1]))
        Jb[:, i] = Adjoint(T) * Blist[:, i]
    return _format_out(sp.trigsimp(Jb))

def JacobianSpace(Slist, thetalist):
    Slist = sp.Matrix(Slist)
    thetalist = sp.Matrix(thetalist)
    Js = sp.Matrix(Slist)
    T = sp.eye(4)
    for i in range(1, len(thetalist)):
        T = T * MatrixExp6(VecTose3(Slist[:, i - 1] * thetalist[i - 1]))
        Js[:, i] = Adjoint(T) * Slist[:, i]
    return _format_out(sp.trigsimp(Js))

def IKinBody(Blist, M, T, thetalist0, eomg, ev):
    Blist = sp.Matrix(Blist)
    M = sp.Matrix(M)
    T = sp.Matrix(T)
    thetalist = sp.Matrix(list(thetalist0)).evalf()

    if (Blist.free_symbols or M.free_symbols or 
        T.free_symbols or thetalist.free_symbols):
        raise ValueError("IKinBody is an iterative numerical solver and does not support symbolic variables.")

    i = 0
    maxiterations = 20
    Vb = sp.Matrix(se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) * T)))
    err = sp.Matrix(Vb[:3]).norm() > eomg or sp.Matrix(Vb[3:6]).norm() > ev

    while err and i < maxiterations:
        Jb = np.array(JacobianBody(Blist, thetalist).evalf(), dtype=float)
        pinv_Jb = sp.Matrix(np.linalg.pinv(Jb))
        thetalist = thetalist + pinv_Jb * Vb
        i += 1
        Vb = sp.Matrix(se3ToVec(MatrixLog6(TransInv(FKinBody(M, Blist, thetalist)) * T)))
        err = sp.Matrix(Vb[:3]).norm() > eomg or sp.Matrix(Vb[3:6]).norm() > ev

    return (_format_out(thetalist), bool(not err))

def IKinSpace(Slist, M, T, thetalist0, eomg, ev):
    Slist = sp.Matrix(Slist)
    M = sp.Matrix(M)
    T = sp.Matrix(T)
    thetalist = sp.Matrix(list(thetalist0)).evalf()

    if (Slist.free_symbols or M.free_symbols or 
        T.free_symbols or thetalist.free_symbols):
        raise ValueError("IKinSpace is an iterative numerical solver and does not support symbolic variables.")

    i = 0
    maxiterations = 20
    Tsb = FKinSpace(M, Slist, thetalist)
    Vs = sp.Matrix(Adjoint(Tsb) * sp.Matrix(se3ToVec(MatrixLog6(TransInv(Tsb) * T))))
    err = sp.Matrix(Vs[:3]).norm() > eomg or sp.Matrix(Vs[3:6]).norm() > ev

    while err and i < maxiterations:
        Js = np.array(JacobianSpace(Slist, thetalist).evalf(), dtype=float)
        pinv_Js = sp.Matrix(np.linalg.pinv(Js))
        thetalist = thetalist + pinv_Js * Vs
        i += 1
        Tsb = FKinSpace(M, Slist, thetalist)
        Vs = sp.Matrix(Adjoint(Tsb) * sp.Matrix(se3ToVec(MatrixLog6(TransInv(Tsb) * T))))
        err = sp.Matrix(Vs[:3]).norm() > eomg or sp.Matrix(Vs[3:6]).norm() > ev

    return (_format_out(thetalist), bool(not err))

def ad(V):
    V = sp.Matrix(list(V))
    omgmat = sp.Matrix(VecToso3(sp.Matrix(V[:3])))
    vmat = sp.Matrix(VecToso3(sp.Matrix(V[3:6])))
    upper = omgmat.row_join(sp.zeros(3, 3))
    lower = vmat.row_join(omgmat)
    return _format_out(upper.col_join(lower))


def InverseDynamics(thetalist, dthetalist, ddthetalist, g, Ftip, Mlist, Glist, Slist):
    thetalist = sp.Matrix(list(thetalist))
    dthetalist = sp.Matrix(list(dthetalist))
    ddthetalist = sp.Matrix(list(ddthetalist))
    g = sp.Matrix(list(g))
    Fi = sp.Matrix(list(Ftip))
    Slist = sp.Matrix(Slist)
    Mlist = [sp.Matrix(M) for M in Mlist]
    Glist = [sp.Matrix(G) for G in Glist]

    n = len(thetalist)
    Mi = sp.eye(4)
    Ai = sp.zeros(6, n)
    AdTi = [None] * (n + 1)
    Vi = sp.zeros(6, n + 1)
    Vdi = sp.zeros(6, n + 1)
    Vdi[:, 0] = sp.Matrix([0, 0, 0]).col_join(-g)
    AdTi[n] = sp.Matrix(Adjoint(TransInv(Mlist[n])))
    taulist = sp.zeros(n, 1)

    for i in range(n):
        Mi = Mi * Mlist[i]
        Ai[:, i] = sp.Matrix(Adjoint(TransInv(Mi))) * Slist[:, i]
        exp_term = sp.Matrix(MatrixExp6(VecTose3(Ai[:, i] * -thetalist[i])))
        inv_M = sp.Matrix(TransInv(Mlist[i]))
        AdTi[i] = sp.Matrix(Adjoint(exp_term * inv_M))
        Vi[:, i + 1] = AdTi[i] * Vi[:, i] + Ai[:, i] * dthetalist[i]
        Vdi[:, i + 1] = (AdTi[i] * Vdi[:, i] 
                         + Ai[:, i] * ddthetalist[i] 
                         + sp.Matrix(ad(Vi[:, i + 1])) * Ai[:, i] * dthetalist[i])

    for i in range(n - 1, -1, -1):
        ad_Vi_T = sp.Matrix(ad(Vi[:, i + 1])).T
        Fi = (AdTi[i + 1].T * Fi 
              + Glist[i] * Vdi[:, i + 1] 
              - ad_Vi_T * (Glist[i] * Vi[:, i + 1]))
        taulist[i, 0] = (Fi.T * Ai[:, i])[0, 0]

    return _format_out(sp.trigsimp(taulist))

def MassMatrix(thetalist, Mlist, Glist, Slist):
    n = len(thetalist)
    M = sp.zeros(n, n)
    for i in range(n):
        ddthetalist = [0] * n
        ddthetalist[i] = 1
        M[:, i] = sp.Matrix(
            InverseDynamics(thetalist, [0] * n, ddthetalist, [0, 0, 0], [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist)
        )
    return _format_out(sp.trigsimp(M))


def VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist):
    n = len(thetalist)
    return InverseDynamics(thetalist, dthetalist, [0] * n, [0, 0, 0], [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist)


def GravityForces(thetalist, g, Mlist, Glist, Slist):
    n = len(thetalist)
    return InverseDynamics(thetalist, [0] * n, [0] * n, g, [0, 0, 0, 0, 0, 0], Mlist, Glist, Slist)


def EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist):
    n = len(thetalist)
    return InverseDynamics(thetalist, [0] * n, [0] * n, [0, 0, 0], Ftip, Mlist, Glist, Slist)


def ForwardDynamics(thetalist, dthetalist, taulist, g, Ftip, Mlist, Glist, Slist):
    taulist = sp.Matrix(list(taulist))
    M = sp.Matrix(MassMatrix(thetalist, Mlist, Glist, Slist))
    c = sp.Matrix(VelQuadraticForces(thetalist, dthetalist, Mlist, Glist, Slist))
    grav = sp.Matrix(GravityForces(thetalist, g, Mlist, Glist, Slist))
    jtf = sp.Matrix(EndEffectorForces(thetalist, Ftip, Mlist, Glist, Slist))
    rhs = taulist - c - grav - jtf

    if M.free_symbols or rhs.free_symbols:
        ddthetalist = M.LUsolve(rhs)
    else:
        ddthetalist = sp.Matrix(np.linalg.solve(np.array(M, dtype=float), np.array(rhs, dtype=float)))

    return _format_out(sp.trigsimp(ddthetalist))

def EulerStep(thetalist, dthetalist, ddthetalist, dt):
    thetalist = sp.Matrix(list(thetalist))
    dthetalist = sp.Matrix(list(dthetalist))
    ddthetalist = sp.Matrix(list(ddthetalist))
    thetalistNext = thetalist + dt * dthetalist
    dthetalistNext = dthetalist + dt * ddthetalist
    return (_format_out(thetalistNext), _format_out(dthetalistNext))


def InverseDynamicsTrajectory(thetamat, dthetamat, ddthetamat, g, Ftipmat, Mlist, Glist, Slist):
    thetamat = sp.Matrix(thetamat).T
    dthetamat = sp.Matrix(dthetamat).T
    ddthetamat = sp.Matrix(ddthetamat).T
    Ftipmat = sp.Matrix(Ftipmat).T
    taumat = sp.zeros(thetamat.rows, thetamat.cols)
    for i in range(thetamat.cols):
        taumat[:, i] = sp.Matrix(
            InverseDynamics(thetamat[:, i], dthetamat[:, i], ddthetamat[:, i], g, Ftipmat[:, i], Mlist, Glist, Slist)
        )
    return _format_out(taumat.T)


def ForwardDynamicsTrajectory(thetalist, dthetalist, taumat, g, Ftipmat, Mlist, Glist, Slist, dt, intRes):
    thetalist = sp.Matrix(list(thetalist))
    dthetalist = sp.Matrix(list(dthetalist))
    taumat = sp.Matrix(taumat).T
    Ftipmat = sp.Matrix(Ftipmat).T
    thetamat = sp.zeros(taumat.rows, taumat.cols)
    dthetamat = sp.zeros(taumat.rows, taumat.cols)
    thetamat[:, 0] = thetalist
    dthetamat[:, 0] = dthetalist
    for i in range(taumat.cols - 1):
        for j in range(intRes):
            ddthetalist = ForwardDynamics(thetalist, dthetalist, taumat[:, i], g, Ftipmat[:, i], Mlist, Glist, Slist)
            thetalist, dthetalist = EulerStep(thetalist, dthetalist, ddthetalist, dt / intRes)
        thetamat[:, i + 1] = sp.Matrix(list(thetalist))
        dthetamat[:, i + 1] = sp.Matrix(list(dthetalist))
    return (_format_out(thetamat.T), _format_out(dthetamat.T))
print(EulerStep([0.1, 0.1, 0.1], [0.1, 0.2, b], [2, 1.5, 1], 0.1))