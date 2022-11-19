from Atomic_Like_Potential import V_eff

# RK4 implementation
def predictor_corrector_radial_shcrodinger_integrator(grid, w10,w20, kwargs):
    E= kwargs['E']
    h= kwargs['delta_x']
    Z= kwargs['Z']
    l= kwargs['l']

    w1=[w10]
    w2=[w20]

    for i,ri in enumerate(grid[:3]):
        k1=4*[0.0]
        k2=4*[0.0]
        vj0= V_eff(ri,Z,l,E)#ALP.V_coulomb(ri, Z) + ALP.V_angular(ri, l)
        vj12= V_eff(ri + 0.5*h,Z,l,E)#ALP.V_coulomb(ri + 0.5*h, Z) + ALP.V_angular(ri + 0.5*h, l)
        vj3= V_eff(ri + h,Z,l,E)#ALP.V_coulomb(ri + h, Z) + ALP.V_angular(ri + h, l)
        for j in range(4):
            #k1[j]=h*(w2[i] + 0.5*k2[j-1])
            if j == 0:
                k1[j]=h*(w2[i])
                k2[j]= h*vj0*w1[i]
            elif j == 1 or j == 2:
                k1[j]=h*(w2[i] + 0.5*k2[j-1])
                k2[j]= h*2.0*vj12*(w1[i] + 0.5*k1[j-1])
            elif j == 3:
                k1[j]=h*(w2[i] + k2[j-1])
                k2[j]= h*2.0*vj3*(w1[i] + k1[j-1])
        w1.append( w1[i] + (1.0/6.0)*(k1[0] + 2.0*k1[1] + 2.0*k1[2] + k1[3]))
        w2.append( w2[i] + (1.0/6.0)*(k2[0] + 2.0*k2[1] + 2.0*k2[2] + k2[3]))

    for i in range(4,len(grid)):
        ri= grid[i]
        r1= grid[i-1]
        r2= grid[i-2]
        r3= grid[i-3]
        r4= grid[i-4]
        wp1= w1[i-1] + h*(55.0*w2[i-1] - 59.0*w2[i-2]
                       +37.0*w2[i-3] - 9.0*w2[i-4])/24.0

        wp2= w2[i-1] + h*(55.0*V_eff(r1,Z,l,E)*w1[i-1] - 59.0*V_eff(r2,Z,l,E)*w1[i-2]
                       +37.0*V_eff(r3,Z,l,E)*w1[i-3] - 9.0*V_eff(r4,Z,l,E)*w1[i-4])/24.0

        wc1= w1[i-1] + h*(9.0*wp2 + 19.0*w2[i-1]
                       -5.0*w2[i-2] + w2[i-3])/24.0

        wc2= w2[i-1] + h*(9.0*V_eff(ri,Z,l,E)*wc1 + 19.0*V_eff(r1,Z,l,E)*w1[i-1]
                       -5.0*V_eff(r2,Z,l,E)*w1[i-2] + V_eff(r3,Z,l,E)*w1[i-3])/24.0

        w1.append(wc1)
        w2.append(wc2)
    return w1


def predictor_corrector_radial_shcrodinger_integrator_p(grid, w10,w20, kwargs):
    E= kwargs['E']
    h= kwargs['delta_x']
    Z= kwargs['Z']
    l= kwargs['l']

    w1=[w10]
    w2=[w20]

    for i,ri in enumerate(grid[:3]):
        k1=4*[0.0]
        k2=4*[0.0]
        vj0= V_eff(ri,Z,l,E)#ALP.V_coulomb(ri, Z) + ALP.V_angular(ri, l)
        vj12= V_eff(ri + 0.5*h,Z,l,E)#ALP.V_coulomb(ri + 0.5*h, Z) + ALP.V_angular(ri + 0.5*h, l)
        vj3= V_eff(ri + h,Z,l,E)#ALP.V_coulomb(ri + h, Z) + ALP.V_angular(ri + h, l)
        for j in range(4):
            #k1[j]=h*(w2[i] + 0.5*k2[j-1])
            if j == 0:
                k1[j]=h*(w2[i])
                k2[j]= h*vj0*w1[i]
            elif j == 1 or j == 2:
                k1[j]=h*(w2[i] + 0.5*k2[j-1])
                k2[j]= h*2.0*vj12*(w1[i] + 0.5*k1[j-1])
            elif j == 3:
                k1[j]=h*(w2[i] + k2[j-1])
                k2[j]= h*2.0*vj3*(w1[i] + k1[j-1])
        w1.append( w1[i] + (1.0/6.0)*(k1[0] + 2.0*k1[1] + 2.0*k1[2] + k1[3]))
        w2.append( w2[i] + (1.0/6.0)*(k2[0] + 2.0*k2[1] + 2.0*k2[2] + k2[3]))

    for i in range(3,len(grid)-2):
        riplus1= grid[i+1]
        ri=grid[i]
        r1= grid[i-1]
        r2= grid[i-2]
        r3= grid[i-3]
        #r4= grid[i-4]
        wp1= w1[i] + h*(55.0*w2[i] - 59.0*w2[i-1]
                       +37.0*w2[i-2] - 9.0*w2[i-3])/24.0

        wp2= w2[i] + h*(55.0*V_eff(ri,Z,l,E)*w1[i] - 59.0*V_eff(r1,Z,l,E)*w1[i-1]
                       +37.0*V_eff(r2,Z,l,E)*w1[i-2] - 9.0*V_eff(r3,Z,l,E)*w1[i-3])/24.0

        wc1= w1[i] + h*(9.0*wp2 + 19.0*w2[i]
                       -5.0*w2[i-1] + w2[i-2])/24.0

        wc2= w2[i] + h*(9.0*V_eff(riplus1,Z,l,E)*wc1 + 19.0*V_eff(r1,Z,l,E)*w1[i]
                       -5.0*V_eff(r1,Z,l,E)*w1[i-1] + V_eff(r2,Z,l,E)*w1[i-2])/24.0

        w1.append(wc1)
        w2.append(wc2)
    return w1