import numpy as np
import modern_robotics
import sympy as sp
from sympy import symbols,diff



class CommonLegCal(object):


    def cal_angular_velocity(self,q_1_list,q_2_list,q_3_list,t):
        length=len(q_1_list)
        # print(q_1_list)
        w_1_list=np.zeros((length,1))
        w_2_list=np.zeros((length,1))
        w_3_list=np.zeros((length,1))
        def extend_array(q_list):
            q_list_extended=np.concatenate((np.array([[q_list[0,0]]]),q_list),axis=0)
            q_list_extended=np.concatenate((q_list_extended,np.array([[q_list[-1,0]]])),axis=0)
            return q_list_extended
        q_1_list=extend_array(q_1_list)
        q_2_list=extend_array(q_2_list)
        q_3_list=extend_array(q_3_list)
        time_interval=t[1,0]-t[0,0]              ############## simple
        for i in range(length):
            w_1_list[i,0]=(q_1_list[i+2,0]-q_1_list[i,0])/(time_interval*2)
            w_2_list[i,0]=(q_2_list[i+2,0]-q_2_list[i,0])/(time_interval*2)
            w_3_list[i,0]=(q_3_list[i+2,0]-q_3_list[i,0])/(time_interval*2)

        # for i in range(length):
        #     if (i<(length-1)):
        #         d_q_1=q_1_list[i+1,0]-q_1_list[i,0]
        #         d_q_2=q_2_list[i+1,0]-q_2_list[i,0]
        #         d_q_3=q_3_list[i+1,0]-q_3_list[i,0]
        #         d_t=t[i+1,0]-t[i,0]
        #         w_1_list[i,0]=d_q_1/d_t
        #         w_2_list[i,0]=d_q_2/d_t
        #         w_3_list[i,0]=d_q_3/d_t
        #     else:
        #         w_1_list[i,0]=w_1_list[i-1,0]
        #         w_2_list[i,0]=w_2_list[i-1,0]
        #         w_3_list[i,0]=w_3_list[i-1,0]
        return w_1_list,w_2_list,w_3_list

    def cal_actuator_mechanical_power(self,torque_1_list,torque_2_list,torque_3_list,w_1_list,w_2_list,w_3_list):
        power_1_list=np.multiply(torque_1_list,w_1_list)
        power_2_list=np.multiply(torque_2_list,w_2_list)
        power_3_list=np.multiply(torque_3_list,w_3_list)
        power_1_list=np.abs(power_1_list)
        power_2_list=np.abs(power_2_list)
        power_3_list=np.abs(power_3_list)
        return power_1_list,power_2_list,power_3_list

    def cal_actuator_mechanical_energy(self,torque_1_list,torque_2_list,torque_3_list,q_1_list,q_2_list,q_3_list):
        length=len(torque_1_list)
        energy_1_list=np.zeros((length,1))
        energy_2_list=np.zeros((length,1))
        energy_3_list=np.zeros((length,1))
        energy_1_now=0
        energy_2_now=0
        energy_3_now=0
        for i in range(length):
            if (i<(length-1)):
                d_q_1=q_1_list[i+1,0]-q_1_list[i,0]
                d_q_2=q_2_list[i+1,0]-q_2_list[i,0]
                d_q_3=q_3_list[i+1,0]-q_3_list[i,0]
                energy_1_now=energy_1_now+np.abs(torque_1_list[i,0]*d_q_1)
                energy_2_now=energy_2_now+np.abs(torque_2_list[i,0]*d_q_2)
                energy_3_now=energy_3_now+np.abs(torque_3_list[i,0]*d_q_3)
                energy_1_list[i,0]=energy_1_now
                energy_2_list[i,0]=energy_2_now
                energy_3_list[i,0]=energy_3_now
            else:
                energy_1_list[i,0]=energy_1_now
                energy_2_list[i,0]=energy_2_now
                energy_3_list[i,0]=energy_3_now
        
        return energy_1_list,energy_2_list,energy_3_list

    def cal_single_leg_MCOT(self,power_1_list,power_2_list,power_3_list,m,v):
        g=9.81
        length=len(power_1_list)
        power_sum=power_1_list.sum()/length+power_2_list.sum()/length+power_3_list.sum()/length
        MCOT=power_sum/(m*g*v)
        return MCOT

    def cal_single_leg_total_energy_loss(self,torque_1_list,torque_2_list,torque_3_list,q_1_list,q_2_list,q_3_list):
        energy_1_list,energy_2_list,energy_3_list=self.cal_actuator_mechanical_energy(torque_1_list,torque_2_list,torque_3_list,q_1_list,q_2_list,q_3_list)
        energy_total=energy_1_list+energy_2_list+energy_3_list
        energy_total_loss=energy_total[-1,0]
        return energy_total_loss

    def cal_single_leg_max_torque(self,torque_1_list,torque_2_list,torque_3_list):
        torque_1_max=np.max(np.abs(torque_1_list))
        torque_2_max=np.max(np.abs(torque_2_list))
        torque_3_max=np.max(np.abs(torque_3_list))
        max_array=np.array([torque_1_max,torque_2_max,torque_3_max])
        max_torque=np.max(max_array)
        return max_torque

    def cal_single_leg_max_angular_velocity(self,w_1_list,w_2_list,w_3_list):
        w_1_max=np.max(np.abs(w_1_list))
        w_2_max=np.max(np.abs(w_2_list))
        w_3_max=np.max(np.abs(w_3_list))
        max_array=np.array([w_1_max,w_2_max,w_3_max])
        max_w=np.max(max_array)
        return max_w

    def cal_single_leg_max_mechanical_power(self,power_1_list,power_2_list,power_3_list):
        power_1_max=np.max(power_1_list)
        power_2_max=np.max(power_2_list)
        power_3_max=np.max(power_3_list)
        max_array=np.array([power_1_max,power_2_max,power_3_max])
        max_power=np.max(max_array)
        return max_power




class BennettLegKin3D(CommonLegCal):

    def __init__(self,b,alpha,beta,L_o,L_s):
        self.b=b
        self.alpha=alpha
        self.beta=beta
        self.L_o=L_o
        self.L_s=L_s
        self.a=b*np.sin(alpha)/np.sin(beta)
        self.K=np.sin(0.5*(beta+alpha))/np.sin(0.5*(beta-alpha))

    def forward_kin_threedof_Bennett(self,q_1,q_2,q_3):
        fai_1=q_1
        fai_2=q_3
        fai_3=-2*np.arctan(-self.K*np.tan((q_3-q_2)/2))

        P_0=[[1, 0, 0, self.a+self.b+self.L_s], [0, 1, 0, 0], [0, 0, 1, self.L_o-0.006], [0, 0, 0, 1]]
        Slist = np.array([[1, 0, 0, 0, 0, 0], 
                        [0, 0, -1, 0, 0, 0], 
                        [0, np.sin(self.alpha),  np.cos(self.alpha), -self.L_o*np.sin(self.alpha), -self.a*np.cos(self.alpha), self.a*np.sin(self.alpha)]]).T  

        failist = [fai_1,fai_2,fai_3]
        T=modern_robotics.FKinSpace(P_0,Slist,failist)
        x_p=T[0,3]
        y_p=T[1,3]
        z_p=T[2,3]
        return x_p,y_p,z_p

    def cal_mapping_torque(self,q_1,q_2,q_3,f_x,f_y,f_z):
        fai_1=q_1
        fai_2=q_3
        fai_3=2*np.arctan(-self.K*np.tan((q_3-q_2)/2))
        Slist = np.array([[1, 0, 0, 0, 0, 0], 
                        [0, 0, -1, 0, 0, 0], 
                        [0, np.sin(self.alpha),  -np.cos(self.alpha), -self.L_o*np.sin(self.alpha), self.a*np.cos(self.alpha), self.a*np.sin(self.alpha)]]).T 
        failist = [fai_1,fai_2,fai_3]
        jacobian_space_mat=modern_robotics.JacobianSpace(Slist,failist)
        jacobian_space_mat=np.mat(jacobian_space_mat)
        def fai_3_func(q_2_s,q_3_s):
            func=2*sp.atan(-self.K*sp.tan((q_3_s-q_2_s)/2))
            return func
        q_2_s,q_3_s=symbols('q_2_s,q_3_s')
        fai_3_diff_q_2=diff(fai_3_func(q_2_s,q_3_s),q_2_s).subs([(q_2_s,q_2),(q_3_s,q_3)])
        fai_3_diff_q_3=diff(fai_3_func(q_2_s,q_3_s),q_3_s).subs([(q_2_s,q_2),(q_3_s,q_3)])
        deriv_mat=np.array([[1 ,0,0],[0,0,1],[0,fai_3_diff_q_2,fai_3_diff_q_3]])
        deriv_mat=np.mat(deriv_mat)
        # print(deriv_mat)
        # print(jacobian_space_mat)
        
        jacobian_q=jacobian_space_mat*deriv_mat
        f=np.array([f_x,f_y,f_z])
        # print(jacobian_q.T)
        x_p,y_p,z_p=self.forward_kin_threedof_Bennett(q_1,q_2,q_3)
        # print(x_p,y_p,z_p)
        r=np.array([x_p,y_p,z_p])
        tor=np.cross(r,f)
        wrench=np.append(tor,f)
        wrench=np.mat(wrench).T
        # print(wrench)
        torque=jacobian_q.T*wrench
        # print(torque)
        torque_1=torque[0]
        torque_2=torque[1]
        torque_3=torque[2]
        return torque_1,torque_2,torque_3

    def __sinCosEquSolve(self,A, B, C):
        '''
        solve A*sin(q)+B*cos(q)+C=0
        '''
        q_1 = np.arctan2(-C, (A**2+B**2-C**2)**0.5) - np.arctan2(B, A)
        q_2 = np.arctan2(-C, -(A**2+B**2-C**2)**0.5) - np.arctan2(B, A)
        q_list = np.array([q_1, q_2])
        for i in range(len(q_list)):
            if q_list[i] > np.pi:
                q_list[i] -= 2*np.pi
            elif q_list[i] < -np.pi:
                q_list[i] += 2*np.pi
        return q_list

    def __checkfai(self,q, q_min, q_max):
        bool_min = q > q_min
        bool_max = q < q_max
        result = np.zeros(len(q))
        for i in range(len(q)):
            if (bool_min[i] and bool_max[i]):
                result[i] = True
            else:
                result[i] = False
        if result.all(): # 两个解
            return q # 1d array
        elif result.any(): # 一个解
            q_new = np.delete(q, np.where(result == False)[0])
            return q_new  # 1d array
        else: # 无解
            #print('Error: There is no fesible solution')
            return [] # 空值 

    def analytical_IK_loss_threedof_Bennett(self,x_now,y_now,z_now):
        px=x_now
        py=y_now
        pz=z_now
        fai_1_min=0
        fai_1_max=np.pi
        eta_lim_max = np.deg2rad(170) # max inter angle of Bennett linkage (q_3-q_1)
        eta_lim_min = np.deg2rad(10) # min inter angle of Bennett linkage (q_3-q_1)
        fai_3_min = 2*np.arctan(-self.K*np.tan((eta_lim_min)/2)) * np.ones(2)
        fai_3_max = 2*np.arctan(-self.K*np.tan((eta_lim_max)/2)) * np.ones(2)
        fai_2_min = np.zeros(2)
        fai_2_max = np.pi*np.ones(2)
        fai_1_min = np.zeros(2)
        fai_1_max = np.pi*np.ones(2)
        fai_list = [] # 所有解

        A1=2*self.L_s*self.L_o*np.sin(self.alpha)+2*self.b*self.L_o*np.sin(self.alpha)
        B1=-2*self.L_s*self.a-2*self.a*self.b
        C1=-self.L_s**2-2*self.L_s*self.b-self.a**2-self.b**2-self.L_o**2+px**2+py**2+pz**2

        fai_3_IK_list_origin=self.__sinCosEquSolve(A1,B1,C1)
        fai_3_IK_list = self.__checkfai(fai_3_IK_list_origin, fai_3_min, fai_3_max)

        if len(fai_3_IK_list)==0:
            return False,False,False


        for fai_3 in fai_3_IK_list:
            A2=-(self.L_s+self.b)*np.sin(fai_3)*np.cos(self.alpha)
            B2=(self.L_s+self.b)*np.cos(fai_3)+self.a
            C2=-px
            fai_2_IK_list_origin = self.__sinCosEquSolve(A2, B2, C2) # fai_2 [0, pi]
            fai_2_IK_list = self.__checkfai(fai_2_IK_list_origin, fai_2_min, fai_2_max)
            #q2_list = q2_origin
            if len(fai_2_IK_list)==0:
                return False,False,False
            for fai_2 in fai_2_IK_list:
                A3=-py
                B3=pz
                C3=self.L_s*np.sin(self.alpha)*np.sin(fai_3)+self.b*np.sin(self.alpha)*np.sin(fai_3)-self.L_o
                fai_1_IK_list_origin = self.__sinCosEquSolve(A3, B3, C3) # fai_1 [0, pi]
                fai_1_IK_list = self.__checkfai(fai_1_IK_list_origin, fai_1_min, fai_1_max)
                if len(fai_1_IK_list)==0:
                    return False,False,False
                for fai_1 in fai_1_IK_list:
                    fai_list.append(np.array([fai_1,fai_2, fai_3]))

        fai_array=fai_list[0]
        q_1=fai_array[0]
        q_3=fai_array[1]
        q_2=q_3-2*np.arctan((-1/self.K)*np.tan(fai_array[2]/2))
        return q_1,q_2,q_3


    def cal_IK_analyical(self,x_b_list,y_b_list,z_b_list):
        # list here do not mean the list type
        # input lengthx1 array
        # output lengthx1 array
        length=len(x_b_list)
        q_1_list=np.zeros((length,1))
        q_2_list=np.zeros((length,1))
        q_3_list=np.zeros((length,1))

        for i in range(length):
            x_now=x_b_list[i,0]
            y_now=y_b_list[i,0]
            z_now=z_b_list[i,0]
            q_1,q_2,q_3=self.analytical_IK_loss_threedof_Bennett(x_now,y_now,z_now)
            q_1_list[i,0]=q_1
            q_2_list[i,0]=q_2
            q_3_list[i,0]=q_3
        return q_1_list,q_2_list,q_3_list

    def verify_IK(self,q_1_list,q_2_list,q_3_list,x_b_list,y_b_list,z_b_list):
        length=len(q_1_list)
        IK_error_num=0
        for i in range(length):
            x_now=x_b_list[i,0]
            y_now=y_b_list[i,0]
            z_now=z_b_list[i,0]
            q_1=q_1_list[i,0]
            q_2=q_2_list[i,0]
            q_3=q_3_list[i,0]
            x_p,y_p,z_p=self.forward_kin_threedof_Bennett(q_1,q_2,q_3)
            error=(x_p-x_now)**2+(y_p-y_now)**2+(z_p-z_now)**2
            if error>=0.01:
                IK_error_num=IK_error_num+1

        return IK_error_num

    def cal_joint_torque_list_threedof_Bennett(self,q_1_list,q_2_list,q_3_list,force_x_list,force_y_list,force_z_list):
        # list here do not mean the list type
        # input lengthx1 array
        # output lengthx1 array
        length=len(q_1_list)
        torque_1_list=np.zeros((length,1))
        torque_2_list=np.zeros((length,1))
        torque_3_list=np.zeros((length,1))
        for i in range(length):
            q_1=q_1_list[i,0]
            q_2=q_2_list[i,0]
            q_3=q_3_list[i,0]
            f_x=force_x_list[i,0]
            f_y=force_y_list[i,0]
            f_z=force_z_list[i,0]
            torque_1,torque_2,torque_3=self.cal_mapping_torque(q_1,q_2,q_3,f_x,f_y,f_z)
            torque_1_list[i,0]=torque_1
            torque_2_list[i,0]=torque_2
            torque_3_list[i,0]=torque_3
        return torque_1_list,torque_2_list,torque_3_list


    def forward_kin_threedof_Bennett_point_1(self,q_1,q_2,q_3):
        fai_1=q_1
        # fai_2=q_3
        # fai_3=2*np.arctan(-self.K*np.tan((q_3-q_2)/2))
        P_0=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.L_o], [0, 0, 0, 1]]
        Slist = np.array([[1, 0, 0, 0, 0, 0]]).T  
        failist = [fai_1]
        T=modern_robotics.FKinSpace(P_0,Slist,failist)
        x_p=T[0,3]
        y_p=T[1,3]
        z_p=T[2,3]
        return x_p,y_p,z_p

    def forward_kin_threedof_Bennett_point_2(self,q_1,q_2,q_3):
        fai_1=q_1
        fai_2=q_3
        fai_3=2*np.arctan(-self.K*np.tan((q_3-q_2)/2))
        P_0=[[1, 0, 0, self.a], [0, 1, 0, 0], [0, 0, 1, self.L_o], [0, 0, 0, 1]]
        Slist = np.array([[1, 0, 0, 0, 0, 0], 
                        [0, 0, -1, 0, 0, 0] ]).T  
        failist = [fai_1,fai_2]
        T=modern_robotics.FKinSpace(P_0,Slist,failist)
        x_p=T[0,3]
        y_p=T[1,3]
        z_p=T[2,3]
        return x_p,y_p,z_p

    def forward_kin_threedof_Bennett_point_4(self,q_1,q_2,q_3):
        fai_1=q_1
        fai_2=q_2
        # fai_3=2*np.arctan(-self.K*np.tan((q_3-q_2)/2))
        P_0=[[1, 0, 0, self.b], [0, 1, 0, 0], [0, 0, 1, self.L_o], [0, 0, 0, 1]]
        Slist = np.array([[1, 0, 0, 0, 0, 0], 
                        [0, 0, -1, 0, 0, 0] ]).T  
        failist = [fai_1,fai_2]
        T=modern_robotics.FKinSpace(P_0,Slist,failist)
        x_p=T[0,3]
        y_p=T[1,3]
        z_p=T[2,3]
        return x_p,y_p,z_p


    def forward_kin_threedof_Bennett_point_3(self,q_1,q_2,q_3):
        fai_1=q_1
        fai_2=q_3
        fai_3=2*np.arctan(-self.K*np.tan((q_3-q_2)/2))
        P_0=[[1, 0, 0, self.a+self.b], [0, 1, 0, 0], [0, 0, 1, self.L_o], [0, 0, 0, 1]]
        Slist = np.array([[1, 0, 0, 0, 0, 0], 
                        [0, 0, -1, 0, 0, 0], 
                        [0, np.sin(self.alpha),  -np.cos(self.alpha), -self.L_o*np.sin(self.alpha), self.a*np.cos(self.alpha), self.a*np.sin(self.alpha)]]).T  
        failist = [fai_1,fai_2,fai_3]
        T=modern_robotics.FKinSpace(P_0,Slist,failist)
        x_p=T[0,3]
        y_p=T[1,3]
        z_p=T[2,3]
        return x_p,y_p,z_p


    def forward_kin_threedof_Bennett_point_5(self,q_1,q_2,q_3):
        fai_1=q_1
        fai_2=q_3
        fai_3=2*np.arctan(-self.K*np.tan((q_3-q_2)/2))
        P_0=[[1, 0, 0, self.a+self.b+self.L_s], [0, 1, 0, 0], [0, 0, 1, self.L_o], [0, 0, 0, 1]]
        Slist = np.array([[1, 0, 0, 0, 0, 0], 
                        [0, 0, -1, 0, 0, 0], 
                        [0, np.sin(self.alpha),  -np.cos(self.alpha), -self.L_o*np.sin(self.alpha), self.a*np.cos(self.alpha), self.a*np.sin(self.alpha)]]).T  
        failist = [fai_1,fai_2,fai_3]
        T=modern_robotics.FKinSpace(P_0,Slist,failist)
        x_p=T[0,3]
        y_p=T[1,3]
        z_p=T[2,3]
        return x_p,y_p,z_p

    
    def ani_joint_points_cal_FL(self,q_1_list,q_2_list,q_3_list):
        ####  1 /\
        #      /  \
        #   2 /    \4
        #     \    /
        #      \  /
        #       \/3
        #        \5
        length=len(q_1_list)
        point_1_list=np.zeros((length,3))
        point_2_list=np.zeros((length,3))
        point_3_list=np.zeros((length,3))
        point_4_list=np.zeros((length,3))
        point_5_list=np.zeros((length,3))
        for i in range(length):
            q_1=q_1_list[i,0]
            q_2=q_2_list[i,0]
            q_3=q_3_list[i,0]
            point_1_x,point_1_y,point_1_z=BennettLegKin3D.forward_kin_threedof_Bennett_point_1(self,q_1,q_2,q_3)
            point_2_x,point_2_y,point_2_z=BennettLegKin3D.forward_kin_threedof_Bennett_point_2(self,q_1,q_2,q_3)
            point_3_x,point_3_y,point_3_z=BennettLegKin3D.forward_kin_threedof_Bennett_point_3(self,q_1,q_2,q_3)
            point_4_x,point_4_y,point_4_z=BennettLegKin3D.forward_kin_threedof_Bennett_point_4(self,q_1,q_2,q_3)
            point_5_x,point_5_y,point_5_z=BennettLegKin3D.forward_kin_threedof_Bennett_point_5(self,q_1,q_2,q_3)

            point_1_list[i,0]=point_1_x
            point_1_list[i,1]=point_1_y
            point_1_list[i,2]=point_1_z

            point_2_list[i,0]=point_2_x
            point_2_list[i,1]=point_2_y
            point_2_list[i,2]=point_2_z

            point_3_list[i,0]=point_3_x
            point_3_list[i,1]=point_3_y
            point_3_list[i,2]=point_3_z

            point_4_list[i,0]=point_4_x
            point_4_list[i,1]=point_4_y
            point_4_list[i,2]=point_4_z

            point_5_list[i,0]=point_5_x
            point_5_list[i,1]=point_5_y
            point_5_list[i,2]=point_5_z

        return point_1_list,point_2_list,point_3_list,point_4_list,point_5_list
    

    # def cal_angular_velocity(self,q_1_list,q_2_list,q_3_list,t):
    #     length=len(q_1_list)
    #     w_1_list=np.zeros((length,1))
    #     w_2_list=np.zeros((length,1))
    #     w_3_list=np.zeros((length,1))
    #     for i in range(length):
    #         if (i<(length-1)):
    #             d_q_1=q_1_list[i+1,0]-q_1_list[i,0]
    #             d_q_2=q_2_list[i+1,0]-q_2_list[i,0]
    #             d_q_3=q_3_list[i+1,0]-q_3_list[i,0]
    #             d_t=t[i+1,0]-t[i,0]
    #             w_1_list[i,0]=d_q_1/d_t
    #             w_2_list[i,0]=d_q_2/d_t
    #             w_3_list[i,0]=d_q_3/d_t
    #         else:
    #             w_1_list[i,0]=w_1_list[i-1,0]
    #             w_2_list[i,0]=w_2_list[i-1,0]
    #             w_3_list[i,0]=w_3_list[i-1,0]
    #     return w_1_list,w_2_list,w_3_list

    # def cal_actuator_mechanical_power(self,torque_1_list,torque_2_list,torque_3_list,w_1_list,w_2_list,w_3_list):
    #     power_1_list=np.multiply(torque_1_list,w_1_list)
    #     power_2_list=np.multiply(torque_2_list,w_2_list)
    #     power_3_list=np.multiply(torque_3_list,w_3_list)
    #     power_1_list=np.abs(power_1_list)
    #     power_2_list=np.abs(power_2_list)
    #     power_3_list=np.abs(power_3_list)
    #     return power_1_list,power_2_list,power_3_list

    # def cal_actuator_mechanical_energy(self,torque_1_list,torque_2_list,torque_3_list,q_1_list,q_2_list,q_3_list):
    #     length=len(torque_1_list)
    #     energy_1_list=np.zeros((length,1))
    #     energy_2_list=np.zeros((length,1))
    #     energy_3_list=np.zeros((length,1))
    #     energy_1_now=0
    #     energy_2_now=0
    #     energy_3_now=0
    #     for i in range(length):
    #         if (i<(length-1)):
    #             d_q_1=q_1_list[i+1,0]-q_1_list[i,0]
    #             d_q_2=q_2_list[i+1,0]-q_2_list[i,0]
    #             d_q_3=q_3_list[i+1,0]-q_3_list[i,0]
    #             energy_1_now=energy_1_now+np.abs(torque_1_list[i,0]*d_q_1)
    #             energy_2_now=energy_2_now+np.abs(torque_2_list[i,0]*d_q_2)
    #             energy_3_now=energy_3_now+np.abs(torque_3_list[i,0]*d_q_3)
    #             energy_1_list[i,0]=energy_1_now
    #             energy_2_list[i,0]=energy_2_now
    #             energy_3_list[i,0]=energy_3_now
    #         else:
    #             energy_1_list[i,0]=energy_1_now
    #             energy_2_list[i,0]=energy_2_now
    #             energy_3_list[i,0]=energy_3_now
        
    #     return energy_1_list,energy_2_list,energy_3_list

    # def cal_single_leg_MCOT(self,power_1_list,power_2_list,power_3_list,m,v):
    #     g=9.81
    #     length=len(power_1_list)
    #     power_sum=power_1_list.sum()/length+power_2_list.sum()/length+power_3_list.sum()/length
    #     MCOT=power_sum/(m*g*v)
    #     return MCOT

    # def cal_single_leg_total_energy_loss(self,torque_1_list,torque_2_list,torque_3_list,q_1_list,q_2_list,q_3_list):
    #     energy_1_list,energy_2_list,energy_3_list=self.cal_actuator_mechanical_energy(torque_1_list,torque_2_list,torque_3_list,q_1_list,q_2_list,q_3_list)
    #     energy_total=energy_1_list+energy_2_list+energy_3_list
    #     energy_total_loss=energy_total[-1,0]
    #     return energy_total_loss

    # def cal_single_leg_max_torque(self,torque_1_list,torque_2_list,torque_3_list):
    #     torque_1_max=np.max(np.abs(torque_1_list))
    #     torque_2_max=np.max(np.abs(torque_2_list))
    #     torque_3_max=np.max(np.abs(torque_3_list))
    #     max_array=np.array([torque_1_max,torque_2_max,torque_3_max])
    #     max_torque=np.max(max_array)
    #     return max_torque

    # def cal_single_leg_max_angular_velocity(self,w_1_list,w_2_list,w_3_list):
    #     w_1_max=np.max(np.abs(w_1_list))
    #     w_2_max=np.max(np.abs(w_2_list))
    #     w_3_max=np.max(np.abs(w_3_list))
    #     max_array=np.array([w_1_max,w_2_max,w_3_max])
    #     max_w=np.max(max_array)
    #     return max_w

    # def cal_single_leg_max_mechanical_power(self,power_1_list,power_2_list,power_3_list):
    #     power_1_max=np.max(power_1_list)
    #     power_2_max=np.max(power_2_list)
    #     power_3_max=np.max(power_3_list)
    #     max_array=np.array([power_1_max,power_2_max,power_3_max])
    #     max_power=np.max(max_array)
    #     return max_power




class PlanarLegKin3D(CommonLegCal):

    def __init__(self,a,b,c,d,L_o,L_s):
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        self.L_o=L_o
        self.L_s=L_s


    def forward_kin_Planar3D(self,q_1,q_2,q_3):
        fai_1=q_1
        fai_2=q_3

        theta_1=q_3-q_2
        f=(self.a**2+self.d**2-2*self.a*self.d*np.cos(theta_1))**0.5
        alpha_1=np.arccos((self.a**2+f**2-self.d**2)/(2*self.a*f))
        alpha_2=np.arccos((self.b**2+f**2-self.c**2)/(2*self.b*f))

        fai_3=np.pi-alpha_1-alpha_2
        # print('fai3:',fai_3*180/np.pi)
        P_0=[[1, 0, 0, self.a+self.b+self.L_s], [0, 1, 0, 0], [0, 0, 1, self.L_o], [0, 0, 0, 1]]

        Slist = np.array([[1, 0, 0, 0, 0, 0], 
                        [0, 0, -1, 0, 0, 0], 
                        [0, 0, 1, 0, -self.a, 0]]).T  

        failist = [fai_1,fai_2,fai_3]
        T=modern_robotics.FKinSpace(P_0,Slist,failist)
        x_p=T[0,3]
        y_p=T[1,3]
        z_p=T[2,3]
        return x_p,y_p,z_p

    def analytical_IK_Planar3D_single(self,x_now,y_now,z_now):
        dis=(y_now**2+z_now**2)**0.5

        fai_1=np.pi-np.arccos(self.L_o/dis)-(np.pi+np.arctan2(y_now,z_now))
        # print(np.arctan2(y_now,z_now)*180/np.pi)
        q_1=fai_1
        T_S_B=np.array([[1,0,0, 0],
                        [0,np.cos(fai_1) ,-np.sin(fai_1),-self.L_o*np.sin(fai_1)],
                        [0,np.sin(fai_1),np.cos(fai_1),self.L_o*np.cos(fai_1)],
                        [0,0,0,1]])
        T_B_S=np.linalg.inv(T_S_B)
        p_b=T_B_S@np.array([[x_now],[y_now],[z_now],[1]])
        # print(p_b)
        X_b=p_b[0]
        Y_b=p_b[1]
        Z_b=p_b[2]
        # print(Z_b)
        f=(X_b**2+Y_b**2)**0.5
        BLS=self.b+self.L_s
        alpha_12=np.arccos((self.a**2+BLS**2-f**2)/(2*self.a*BLS))
        e=(self.a**2+self.b**2-2*self.a*self.b*np.cos(alpha_12))**0.5
        beta_2=np.arctan2(Y_b,X_b)
        # print('beta_2',beta_2*180/np.pi)
        beta_1=np.arccos((self.a**2+f**2-BLS**2)/(2*self.a*f))
        q_3=-beta_2+beta_1
        gama_1=np.arccos((self.a**2+e**2-self.b**2)/(2*self.a*e))
        gama_2=np.arccos((self.d**2+e**2-self.c**2)/(2*self.d*e))
        q_2=q_3-gama_1-gama_2
        return q_1,q_2,q_3

    def cal_IK_analyical(self,x_b_list,y_b_list,z_b_list):
        # list here do not mean the list type
        # input lengthx1 array
        # output lengthx1 array
        length=len(x_b_list)
        q_1_list=np.zeros((length,1))
        q_2_list=np.zeros((length,1))
        q_3_list=np.zeros((length,1))

        for i in range(length):
            x_now=x_b_list[i,0]
            y_now=y_b_list[i,0]
            z_now=z_b_list[i,0]
            q_1,q_2,q_3=self.analytical_IK_Planar3D_single(x_now,y_now,z_now)
            q_1_list[i,0]=q_1
            q_2_list[i,0]=q_2
            q_3_list[i,0]=q_3
        return q_1_list,q_2_list,q_3_list

    def verify_IK(self,q_1_list,q_2_list,q_3_list,x_b_list,y_b_list,z_b_list):
        length=len(q_1_list)
        IK_error_num=0
        for i in range(length):
            x_now=x_b_list[i,0]
            y_now=y_b_list[i,0]
            z_now=z_b_list[i,0]
            q_1=q_1_list[i,0]
            q_2=q_2_list[i,0]
            q_3=q_3_list[i,0]
            x_p,y_p,z_p=self.forward_kin_Planar3D(q_1,q_2,q_3)
            error=(x_p-x_now)**2+(y_p-y_now)**2+(z_p-z_now)**2
            if error>=0.01:
                IK_error_num=IK_error_num+1

        return IK_error_num

    def cal_mapping_torque(self,q_1,q_2,q_3,f_x,f_y,f_z):
        fai_1=q_1
        fai_2=q_3

        theta_1=q_3-q_2
        f=(self.a**2+self.d**2-2*self.a*self.d*np.cos(theta_1))**0.5
        alpha_1=np.arccos((self.a**2+f**2-self.d**2)/(2*self.a*f))
        alpha_2=np.arccos((self.b**2+f**2-self.c**2)/(2*self.b*f))
        fai_3=np.pi-alpha_1-alpha_2
        # print('fai3:',fai_3*180/np.pi)
        P_0=[[1, 0, 0, self.a+self.b+self.L_s], [0, 1, 0, 0], [0, 0, 1, self.L_o], [0, 0, 0, 1]]

        Slist = np.array([[1, 0, 0, 0, 0, 0], 
                        [0, 0, -1, 0, 0, 0], 
                        [0, 0, 1, 0, -self.a, 0]]).T  

        failist = [fai_1,fai_2,fai_3]

        jacobian_space_mat=modern_robotics.JacobianSpace(Slist,failist)
        jacobian_space_mat=np.mat(jacobian_space_mat)
        def fai_3_func(q_2_s,q_3_s):
            theta_1=q_3_s-q_2_s
            f=(self.a**2+self.d**2-2*self.a*self.d*sp.cos(theta_1))**0.5
            alpha_1=sp.acos((self.a**2+f**2-self.d**2)/(2*self.a*f))
            alpha_2=sp.acos((self.b**2+f**2-self.c**2)/(2*self.b*f))
            func=sp.pi-alpha_1-alpha_2
            return func
        q_2_s,q_3_s=symbols('q_2_s,q_3_s')
        fai_3_diff_q_2=diff(fai_3_func(q_2_s,q_3_s),q_2_s).subs([(q_2_s,q_2),(q_3_s,q_3)])
        fai_3_diff_q_3=diff(fai_3_func(q_2_s,q_3_s),q_3_s).subs([(q_2_s,q_2),(q_3_s,q_3)])
        deriv_mat=np.array([[1 ,0,0],[0,0,1],[0,fai_3_diff_q_2,fai_3_diff_q_3]])
        deriv_mat=np.mat(deriv_mat)
        # print(deriv_mat)
        # print(jacobian_space_mat)
        
        jacobian_q=jacobian_space_mat*deriv_mat
        f=np.array([f_x,f_y,f_z])
        # print(jacobian_q.T)
        x_p,y_p,z_p=self.forward_kin_Planar3D(q_1,q_2,q_3)
        # print(x_p,y_p,z_p)
        r=np.array([x_p,y_p,z_p])
        tor=np.cross(r,f)
        wrench=np.append(tor,f)
        wrench=np.mat(wrench).T
        # print(wrench)
        torque=jacobian_q.T*wrench
        # print(torque)
        torque_1=torque[0]
        torque_2=torque[1]
        torque_3=torque[2]
        return torque_1,torque_2,torque_3

    def cal_joint_torque_list_Planar3D(self,q_1_list,q_2_list,q_3_list,force_x_list,force_y_list,force_z_list):
        # list here do not mean the list type
        # input lengthx1 array
        # output lengthx1 array
        length=len(q_1_list)
        torque_1_list=np.zeros((length,1))
        torque_2_list=np.zeros((length,1))
        torque_3_list=np.zeros((length,1))
        for i in range(length):
            q_1=q_1_list[i,0]
            q_2=q_2_list[i,0]
            q_3=q_3_list[i,0]
            f_x=force_x_list[i,0]
            f_y=force_y_list[i,0]
            f_z=force_z_list[i,0]
            torque_1,torque_2,torque_3=self.cal_mapping_torque(q_1,q_2,q_3,f_x,f_y,f_z)
            torque_1_list[i,0]=torque_1
            torque_2_list[i,0]=torque_2
            torque_3_list[i,0]=torque_3
        return torque_1_list,torque_2_list,torque_3_list




class SerialLegKin3D(CommonLegCal):

    def __init__(self,a,b,c,d,L_o,L_s):
        self.a=a
        self.b=b
        self.c=c
        self.d=d
        self.L_o=L_o
        self.L_s=L_s


    def forward_kin_Serial3D(self,q_1,q_2,q_3):
        fai_1=q_1
        fai_2=q_3

        theta_1=q_3-q_2
        f=(self.a**2+self.d**2-2*self.a*self.d*np.cos(theta_1))**0.5
        alpha_1=np.arccos((self.a**2+f**2-self.d**2)/(2*self.a*f))
        alpha_2=np.arccos((self.b**2+f**2-self.c**2)/(2*self.b*f))

        fai_3=np.pi-alpha_1-alpha_2
        # print('fai3:',fai_3*180/np.pi)
        P_0=[[1, 0, 0, self.a+self.b+self.L_s], [0, 1, 0, 0], [0, 0, 1, self.L_o], [0, 0, 0, 1]]

        Slist = np.array([[1, 0, 0, 0, 0, 0], 
                        [0, 0, -1, 0, 0, 0], 
                        [0, 0, 1, 0, -self.a, 0]]).T  

        failist = [fai_1,fai_2,fai_3]
        T=modern_robotics.FKinSpace(P_0,Slist,failist)
        x_p=T[0,3]
        y_p=T[1,3]
        z_p=T[2,3]
        return x_p,y_p,z_p

    def analytical_IK_Serial3D_single(self,x_now,y_now,z_now):
        dis=(y_now**2+z_now**2)**0.5

        fai_1=np.pi-np.arccos(self.L_o/dis)-(np.pi+np.arctan2(y_now,z_now))
        # print(np.arctan2(y_now,z_now)*180/np.pi)
        q_1=fai_1
        T_S_B=np.array([[1,0,0, 0],
                        [0,np.cos(fai_1) ,-np.sin(fai_1),-self.L_o*np.sin(fai_1)],
                        [0,np.sin(fai_1),np.cos(fai_1),self.L_o*np.cos(fai_1)],
                        [0,0,0,1]])
        T_B_S=np.linalg.inv(T_S_B)
        p_b=T_B_S@np.array([[x_now],[y_now],[z_now],[1]])
        # print(p_b)
        X_b=p_b[0]
        Y_b=p_b[1]
        Z_b=p_b[2]
        # print(Z_b)
        f=(X_b**2+Y_b**2)**0.5
        BLS=self.b+self.L_s
        alpha_12=np.arccos((self.a**2+BLS**2-f**2)/(2*self.a*BLS))
        e=(self.a**2+self.b**2-2*self.a*self.b*np.cos(alpha_12))**0.5
        beta_2=np.arctan2(Y_b,X_b)
        # print('beta_2',beta_2*180/np.pi)
        beta_1=np.arccos((self.a**2+f**2-BLS**2)/(2*self.a*f))
        q_3=-beta_2+beta_1
        gama_1=np.arccos((self.a**2+e**2-self.b**2)/(2*self.a*e))
        gama_2=np.arccos((self.d**2+e**2-self.c**2)/(2*self.d*e))
        q_2=q_3-gama_1-gama_2
        return q_1,q_2,q_3

    def cal_IK_analyical(self,x_b_list,y_b_list,z_b_list):
        # list here do not mean the list type
        # input lengthx1 array
        # output lengthx1 array
        length=len(x_b_list)
        q_1_list=np.zeros((length,1))
        q_2_list=np.zeros((length,1))
        q_3_list=np.zeros((length,1))

        for i in range(length):
            x_now=x_b_list[i,0]
            y_now=y_b_list[i,0]
            z_now=z_b_list[i,0]
            q_1,q_2,q_3=self.analytical_IK_Serial3D_single(x_now,y_now,z_now)
            q_1_list[i,0]=q_1
            q_2_list[i,0]=q_2
            q_3_list[i,0]=q_3
        return q_1_list,q_2_list,q_3_list

    def verify_IK(self,q_1_list,q_2_list,q_3_list,x_b_list,y_b_list,z_b_list):
        length=len(q_1_list)
        IK_error_num=0
        for i in range(length):
            x_now=x_b_list[i,0]
            y_now=y_b_list[i,0]
            z_now=z_b_list[i,0]
            q_1=q_1_list[i,0]
            q_2=q_2_list[i,0]
            q_3=q_3_list[i,0]
            x_p,y_p,z_p=self.forward_kin_Serial3D(q_1,q_2,q_3)
            error=(x_p-x_now)**2+(y_p-y_now)**2+(z_p-z_now)**2
            if error>=0.01:
                IK_error_num=IK_error_num+1

        return IK_error_num

    def cal_mapping_torque(self,q_1,q_2,q_3,f_x,f_y,f_z):
        fai_1=q_1
        fai_2=q_3

        theta_1=q_3-q_2
        f=(self.a**2+self.d**2-2*self.a*self.d*np.cos(theta_1))**0.5
        alpha_1=np.arccos((self.a**2+f**2-self.d**2)/(2*self.a*f))
        alpha_2=np.arccos((self.b**2+f**2-self.c**2)/(2*self.b*f))
        fai_3=np.pi-alpha_1-alpha_2
        # print('fai3:',fai_3*180/np.pi)
        P_0=[[1, 0, 0, self.a+self.b+self.L_s], [0, 1, 0, 0], [0, 0, 1, self.L_o], [0, 0, 0, 1]]

        Slist = np.array([[1, 0, 0, 0, 0, 0], 
                        [0, 0, -1, 0, 0, 0], 
                        [0, 0, 1, 0, -self.a, 0]]).T  

        failist = [fai_1,fai_2,fai_3]

        jacobian_space_mat=modern_robotics.JacobianSpace(Slist,failist)
        jacobian_space_mat=np.mat(jacobian_space_mat)
        def fai_3_func(q_2_s,q_3_s):
            theta_1=q_3_s-q_2_s
            f=(self.a**2+self.d**2-2*self.a*self.d*sp.cos(theta_1))**0.5
            alpha_1=sp.acos((self.a**2+f**2-self.d**2)/(2*self.a*f))
            alpha_2=sp.acos((self.b**2+f**2-self.c**2)/(2*self.b*f))
            func=sp.pi-alpha_1-alpha_2
            return func
        q_2_s,q_3_s=symbols('q_2_s,q_3_s')
        fai_3_diff_q_2=diff(fai_3_func(q_2_s,q_3_s),q_2_s).subs([(q_2_s,q_2),(q_3_s,q_3)])
        fai_3_diff_q_3=diff(fai_3_func(q_2_s,q_3_s),q_3_s).subs([(q_2_s,q_2),(q_3_s,q_3)])
        deriv_mat=np.array([[1 ,0,0],[0,0,1],[0,fai_3_diff_q_2,fai_3_diff_q_3]])
        deriv_mat=np.mat(deriv_mat)
        # print(deriv_mat)
        # print(jacobian_space_mat)
        
        jacobian_q=jacobian_space_mat*deriv_mat
        f=np.array([f_x,f_y,f_z])
        # print(jacobian_q.T)
        x_p,y_p,z_p=self.forward_kin_Serial3D(q_1,q_2,q_3)
        # print(x_p,y_p,z_p)
        r=np.array([x_p,y_p,z_p])
        tor=np.cross(r,f)
        wrench=np.append(tor,f)
        wrench=np.mat(wrench).T
        # print(wrench)
        torque=jacobian_q.T*wrench
        # print(torque)
        torque_1=torque[0]
        torque_2=torque[1]
        torque_3=torque[2]
        return torque_1,torque_2,torque_3

    def cal_joint_torque_list_Serial3D(self,q_1_list,q_2_list,q_3_list,force_x_list,force_y_list,force_z_list):
        # list here do not mean the list type
        # input lengthx1 array
        # output lengthx1 array
        length=len(q_1_list)
        torque_1_list=np.zeros((length,1))
        torque_2_list=np.zeros((length,1))
        torque_3_list=np.zeros((length,1))
        for i in range(length):
            q_1=q_1_list[i,0]
            q_2=q_2_list[i,0]
            q_3=q_3_list[i,0]
            f_x=force_x_list[i,0]
            f_y=force_y_list[i,0]
            f_z=force_z_list[i,0]
            torque_1,torque_2,torque_3=self.cal_mapping_torque(q_1,q_2,q_3,f_x,f_y,f_z)
            torque_1_list[i,0]=torque_1
            torque_2_list[i,0]=torque_2
            torque_3_list[i,0]=torque_3
        return torque_1_list,torque_2_list,torque_3_list
