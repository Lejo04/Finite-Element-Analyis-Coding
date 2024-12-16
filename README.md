# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp

#Input file path (File path needs to be change accordingly otherwise FileNotFoundError will occur
#The input file used for this code is "Abaqus.txt" and was submitted with the python script)
inputfile='C:\\Users\sikaj\\Desktop\\FEACodingProject\\AbaqusInp.txt';
#Coordinate definition
n=sp.symbols('n') # horizontal direction;
e=sp.symbols('e') # vertical direction;

def fea(inputfile):
    #Material propretires
    Y=MaterialProperties(inputfile)[0] #Young modulus
    P=MaterialProperties(inputfile)[1] #Poison ratio
    #Boundary condition
    boundary=Boundarycondition(inputfile)
    # Loading condition
    loading=Load(inputfile)
    # E matrix for plane strain element
    E=((Y)/((1+P)*(1-2*P)))*sp.Matrix([[1-P,P,0],[P,1-P,0],[0,0,(1-2*P)/2]])
    #Global element nod e coordinates
    Node=NodeElement(inputfile)
    #Element connectivity
    Connectivity=ElementConnectivity(inputfile)
    #Shape functions
    N1=(1/4)*(n-1)*(e-1);
    N2=(-1/4)*(n+1)*(e-1);
    N3=(1/4)*(n+1)*(e+1);
    N4=(-1/4)*(n-1)*(e+1);
    # Mapping global element to parent element
    #Element 1
    x_1=(Node[0,0]*N1+Node[1,0]*N2+Node[5,0]*N3+Node[4,0]*N4)
    y_1=(Node[0,1]*N1+Node[1,1]*N2+Node[5,1]*N3+Node[4,1]*N4)
    #Element 2
    x_2=(Node[1,0]*N1+Node[2,0]*N2+Node[6,0]*N3+Node[5,0]*N4)
    y_2=(Node[1,1]*N1+Node[2,1]*N2+Node[6,1]*N3+Node[5,1]*N4)
    #Element 3
    x_3=(Node[2,0]*N1+Node[3,0]*N2+Node[9,0]*N3+Node[6,0]*N4)
    y_3=(Node[2,1]*N1+Node[3,1]*N2+Node[9,1]*N3+Node[6,1]*N4)
    #Element 4
    x_4=(Node[4,0]*N1+Node[5,0]*N2+Node[8,0]*N3+Node[7,0]*N4)
    y_4=(Node[4,1]*N1+Node[5,1]*N2+Node[8,1]*N3+Node[7,1]*N4)
    #ELement 5
    x_5=(Node[5,0]*N1+Node[6,0]*N2+Node[9,0]*N3+Node[8,0]*N4)
    y_5=(Node[5,1]*N1+Node[6,1]*N2+Node[9,1]*N3+Node[8,1]*N4)
    #Computation of the Jacobian
    #Element 1
    J_1=sp.Matrix([[sp.diff(x_1,n), sp.diff(y_1,n)],[sp.diff(x_1,e),sp.diff(y_1,e)]])
    #Element 2
    J_2=sp.Matrix([[sp.diff(x_2,n), sp.diff(y_2,n)],[sp.diff(x_2,e),sp.diff(y_2,e)]])
    #Element 3
    J_3=sp.Matrix([[sp.diff(x_3,n), sp.diff(y_3,n)],[sp.diff(x_3,e),sp.diff(y_3,e)]])
    #Element 4
    J_4=sp.Matrix([[sp.diff(x_4,n), sp.diff(y_4,n)],[sp.diff(x_4,e),sp.diff(y_4,e)]])
    #Element 5
    J_5=sp.Matrix([[sp.diff(x_5,n), sp.diff(y_5,n)],[sp.diff(x_5,e),sp.diff(y_5,e)]])
    #Element 1 B_matrix
    B_1=sp.Matrix([[Chaine_rule(N1,J_1)[0],0.0,Chaine_rule(N2,J_1)[0],0.0,Chaine_rule(N3,J_1)[0],0.0,Chaine_rule(N4,J_1)[0],0.0],[0.0,Chaine_rule(N1,J_1)[1],0.0,Chaine_rule(N2,J_1)[1],0.0,Chaine_rule(N3,J_1)[1],0.0,Chaine_rule(N4,J_1)[1]],
                   [Chaine_rule(N1,J_1)[1],Chaine_rule(N1,J_1)[0],Chaine_rule(N2,J_1)[1],Chaine_rule(N2,J_1)[0],Chaine_rule(N3,J_1)[1],Chaine_rule(N3,J_1)[0],Chaine_rule(N4,J_1)[1],Chaine_rule(N4,J_1)[0]]])
    #Element 2 B_matrix
    B_2=sp.Matrix([[Chaine_rule(N1,J_2)[0],0.0,Chaine_rule(N2,J_2)[0],0.0,Chaine_rule(N3,J_2)[0],0.0,Chaine_rule(N4,J_2)[0],0.0],[0.0,Chaine_rule(N1,J_2)[1],0.0,Chaine_rule(N2,J_2)[1],0.0,Chaine_rule(N3,J_2)[1],0.0,Chaine_rule(N4,J_2)[1]],
                   [Chaine_rule(N1,J_2)[1],Chaine_rule(N1,J_2)[0],Chaine_rule(N2,J_2)[1],Chaine_rule(N2,J_2)[0],Chaine_rule(N3,J_2)[1],Chaine_rule(N3,J_2)[0],Chaine_rule(N4,J_2)[1],Chaine_rule(N4,J_1)[0]]])
    #Element 3 B_matrix
    B_3=B_1=sp.Matrix([[Chaine_rule(N1,J_3)[0],0.0,Chaine_rule(N2,J_3)[0],0.0,Chaine_rule(N3,J_3)[0],0.0,Chaine_rule(N4,J_3)[0],0.0],[0.0,Chaine_rule(N1,J_3)[1],0.0,Chaine_rule(N2,J_3)[1],0.0,Chaine_rule(N3,J_3)[1],0.0,Chaine_rule(N4,J_3)[1]],
                   [Chaine_rule(N1,J_3)[1],Chaine_rule(N1,J_3)[0],Chaine_rule(N2,J_3)[1],Chaine_rule(N2,J_3)[0],Chaine_rule(N3,J_3)[1],Chaine_rule(N3,J_3)[0],Chaine_rule(N4,J_3)[1],Chaine_rule(N4,J_3)[0]]])
    #Element 4 B_matrix
    B_4=sp.Matrix([[Chaine_rule(N1,J_4)[0],0.0,Chaine_rule(N2,J_4)[0],0.0,Chaine_rule(N3,J_4)[0],0.0,Chaine_rule(N4,J_4)[0],0.0],[0.0,Chaine_rule(N1,J_4)[1],0.0,Chaine_rule(N2,J_4)[1],0.0,Chaine_rule(N3,J_4)[1],0.0,Chaine_rule(N4,J_4)[1]],
                   [Chaine_rule(N1,J_4)[1],Chaine_rule(N1,J_4)[0],Chaine_rule(N2,J_4)[1],Chaine_rule(N2,J_4)[0],Chaine_rule(N3,J_4)[1],Chaine_rule(N3,J_4)[0],Chaine_rule(N4,J_4)[1],Chaine_rule(N4,J_4)[0]]])
    #Element 5 B_matrix
    B_5=sp.Matrix([[Chaine_rule(N1,J_5)[0],0.0,Chaine_rule(N2,J_5)[0],0.0,Chaine_rule(N3,J_5)[0],0.0,Chaine_rule(N4,J_5)[0],0.0],[0.0,Chaine_rule(N1,J_5)[1],0.0,Chaine_rule(N2,J_5)[1],0.0,Chaine_rule(N3,J_5)[1],0.0,Chaine_rule(N4,J_5)[1]],
                   [Chaine_rule(N1,J_5)[1],Chaine_rule(N1,J_5)[0],Chaine_rule(N2,J_1)[1],Chaine_rule(N2,J_5)[0],Chaine_rule(N3,J_5)[1],Chaine_rule(N3,J_5)[0],Chaine_rule(N4,J_5)[1],Chaine_rule(N4,J_5)[0]]])
    #Stiffness matrix computation for element 1
    K_1=LocalstiffnessMatrix(B_1, E, J_1)
    # #Stiffness matrix computation for element 2
    K_2=LocalstiffnessMatrix(B_2, E, J_2)
    # #Stiffness matrix computation for element 3
    K_3=LocalstiffnessMatrix(B_3, E, J_3)
    # #Stiffness matrix computation for element 4
    K_4=LocalstiffnessMatrix(B_4, E, J_4)
    #Stiffness matrix computation for element 5
    K_5=LocalstiffnessMatrix(B_5, E, J_5)
    #Global stiffness matrix
    Global_stiffness_matrix=5*0.001*GlobalstiffnessMatrix(Connectivity, K_1, K_2, K_3, K_4, K_5)
    # #Reduced Residual force
    Residual_force_reduced=reducedresidualloadvector(loading, Global_stiffness_matrix, Connectivity, boundary)
    # #Reduced Global stiffness matrix
    Globalstiffness_reduced=reducedGlobalstiffnessMatrix(boundary, Global_stiffness_matrix, Connectivity)
    # #reduced displacement vector
    reduced_displacement =np.linalg.solve(Globalstiffness_reduced,Residual_force_reduced)
    #Displacement at nodes
    nodal_displacement=nodaldisplacement(boundary, Global_stiffness_matrix, Connectivity, reduced_displacement)
    #Reaction force at nodes
    nodal_reaction_force=reactionforce(Global_stiffness_matrix,nodal_displacement)
    #Element A stress at nodal points (1/sqrt(3),1/sqrt(3))
    [s11A,s22A]=StressCalculation(E, B_3,4,Connectivity, nodal_displacement,1/np.sqrt(3), 1/np.sqrt(3))
    #Element A stress at nodal points (-1/sqrt(3),1/sqrt(3))
    [s11B,s22B]=StressCalculation(E, B_3,4,Connectivity, nodal_displacement,-1/np.sqrt(3), 1/np.sqrt(3))
    #Element A stress at nodal points (-1/sqrt(3),-1/sqrt(3))
    [s11C,s22C]=StressCalculation(E, B_3,4,Connectivity, nodal_displacement,-1/np.sqrt(3), -1/np.sqrt(3))
    #Element A stress at nodal points (1/sqrt(3),-1/sqrt(3))
    [s11D,s22D]=StressCalculation(E, B_3,4,Connectivity, nodal_displacement,1/np.sqrt(3), -1/np.sqrt(3))   
    #Nodal displacement output
    print("\n**Nodal displacement output**\n")
    print("Node 1: x-direction=",nodal_displacement[0],"m ", "y-direction=",nodal_displacement[1],"m")
    print("Node 2: x-direction=",nodal_displacement[2],"m ", "y-direction=",nodal_displacement[3],"m")
    print("Node 3: x-direction=",nodal_displacement[8],"m ", "y-direction=",nodal_displacement[9],"m")
    print("Node 4: x-direction=",nodal_displacement[12],"m ", "y-direction=",nodal_displacement[13],"m")
    print("Node 5: x-direction=",nodal_displacement[6],"m ", "y-direction=",nodal_displacement[7],"m")
    print("Node 6: x-direction=",nodal_displacement[4],"m ", "y-direction=",nodal_displacement[5],"m")
    print("Node 7: x-direction=",nodal_displacement[10],"m ", "y-direction=",nodal_displacement[11],"m")
    print("Node 8: x-direction=",nodal_displacement[18],"m ", "y-direction=",nodal_displacement[19],"m")
    print("Node 9: x-direction=",nodal_displacement[16],"m ", "y-direction=",nodal_displacement[17],"m")
    print("Node 10: x-direction=",nodal_displacement[14],"m ", "y-direction=",nodal_displacement[15],"m")
    print("\n")
    # Reaction force at imposed boundary nodes
    print("\n**Reaction force at imposed boundary nodes**\n")
    print("Node 1: x-direction=",nodal_reaction_force[0],"N ", "y-direction=",nodal_reaction_force[1],"N")
    print("Node 2: x-direction=",nodal_reaction_force[2],"N ", "y-direction=",nodal_reaction_force[3],"N")
    print("Node 3: x-direction=",nodal_reaction_force[8],"N ", "y-direction=",nodal_reaction_force[9],"N")
    print("Node 4: x-direction=",nodal_reaction_force[12],"N ", "y-direction=",nodal_reaction_force[13],"N")
    print("Node 5: x-direction=",nodal_reaction_force[6],"N ", "y-direction=",nodal_reaction_force[7],"N")
    print("Node 8: x-direction=",nodal_reaction_force[18],"N ", "y-direction=",nodal_reaction_force[19],"N")
    print("Node 9: x-direction=",nodal_reaction_force[16],"N ", "y-direction=",nodal_reaction_force[17],"N")
    print("Node 10: x-direction=",nodal_reaction_force[14],"N ", "y-direction=",nodal_reaction_force[15],"N")
    print("\n")
    #Computing sig_11 and sig_22 at the four integration point of element A
    print("\n**stress sig_11 and sig_22 at four integration points of element A**\n")
    print("sig__11 at (1/sqrt(3),1/sart(3))= ",s11A," Pa")
    print("sig__22 at (1/sqrt(3),1/sart(3))= ",s22A," Pa")
    print("sig__11 at (-1/sqrt(3),1/sart(3))= ",s11B," Pa")
    print("sig__22 at (-1/sqrt(3),1/sart(3))= ",s22B," Pa")
    print("sig__11 at (-1/sqrt(3),-1/sart(3))= ",s11C," Pa")
    print("sig__22 at (-1/sqrt(3),-1/sart(3))= ",s22C," Pa")
    print("sig__11 at (1/sqrt(3),-1/sart(3))= ",s11D," Pa")
    print("sig__22 at (1/sqrt(3),-1/sart(3))= ",s22D," Pa")
    return nodal_reaction_force

def LocalstiffnessMatrix(B,E,J):
    a=1/np.sqrt(3)
    b=-1/np.sqrt(3)
    K_prime=B.T*E*B*J.det()
    for i in range(K_prime.rows):
        for j in range(K_prime.cols):
            K_prime[i,j]=K_prime[i,j].subs({n:a,e:a})+K_prime[i,j].subs({n:b,e:a})
            +K_prime[i,j].subs({n:b,e:b})
            +K_prime[i,j].subs({n:a,e:b})  
    return K_prime  
  
def GlobalstiffnessMatrix(connectivity,K_1,K_2,K_3,K_4,K_5):
    K_prime=np.zeros((20,20))
    mapping=Node_mapping(connectivity)
    for i_1 in range(K_1.rows):
        for j_1 in range(K_1.cols):
            K_prime[i_1,j_1]=float(K_1[i_1,j_1])
    row_index=1
    col_index=0
    while row_index<mapping.shape[0]:
        stiffness=switch_case(row_index, K_1, K_2, K_3, K_4, K_5)
        tracker=0
        helper=0
        iterator_1=0
        while col_index<mapping.shape[1]:
            a=int(mapping[row_index,col_index])
            iterator_2=iterator_1
            while(iterator_2<stiffness.shape[1]):
                helper=int(mapping[row_index,col_index+tracker])
                K_prime[a,helper]=np.add(float(K_prime[a,helper]),float(stiffness[iterator_1,iterator_2]))
                iterator_2=iterator_2+1
                tracker=tracker+1
            iterator_2=iterator_1+1
            tracker=1
            while(iterator_2<stiffness.shape[0]):
                helper=int(mapping[row_index,col_index+tracker])
                K_prime[helper,a]=np.add(float(K_prime[helper,a]),float(stiffness[iterator_2,iterator_1]))
                iterator_2=iterator_2+1
                tracker=tracker+1
            iterator_1=iterator_1+1
            col_index=col_index+1
            tracker=0
        row_index=row_index+1
        col_index=0
    return K_prime

def reducedGlobalstiffnessMatrix(boundary,K,connectivity):
    node=nodehavingdisplacementboundarycondition(boundary, K, connectivity)
    keeper=np.zeros((K.shape[0],K.shape[1]))
    for a in range (keeper.shape[0]):
        for b in range (keeper.shape[1]):
            keeper[a][b]=K[a][b]
    for i_1 in range(node.shape[0]):
        n=int(node[i_1])
        for i_2 in range(keeper.shape[1]):
            keeper[n,i_2]='-1'
        for i_3 in range(keeper.shape[0]):
            keeper[i_3,n]='-1'
    reduced_matrix=np.zeros((keeper.shape[0]-node.shape[0],keeper.shape[1]-node.shape[0]))
    row_index=0
    change=False
    for i in range(keeper.shape[0]):
        col_index=0
        if change:
            change=False
            row_index=row_index+1
        for j in range(keeper.shape[1]):
            if keeper[i,j]!=-1:
                change=True
                reduced_matrix[row_index,col_index]=keeper[i,j]
                col_index=col_index+1
    return reduced_matrix

def Node_mapping(connectivity):
    position=np.zeros((5,4))
    index=1
    while index<position.shape[1]:
        position[:,index]=position[:,index-1]+2
        index=index+1
    a=1
    b=0
    current_node=0
    previous_element=0
    same_element=True
    while (a<connectivity.shape[0]):
        previous_element=previous_element+1
        while(b<connectivity.shape[1]):
            current_node=connectivity[a,b]
            if ExisitingNode(connectivity, previous_element, current_node)[1]>0:
                position[a,b]=position[ExisitingNode(connectivity, previous_element, current_node)[0],
                                       ExisitingNode(connectivity, previous_element, current_node)[1]]
            else:
                if same_element:
                    if a<=2 :
                        position[a,b]=4*(a+b)
                    else:
                        position[a,b]=4*(a+b)-4
                    same_element=False
                else:
                    position[a,b]=position[a,b-1]+2
            b=b+1
        a=a+1
        b=0
        same_element=True
    index=0
    index_2=0
    final_form=np.zeros((5,8))
    while(index<final_form.shape[0]):
        iterator=0
        while (index_2<=final_form.shape[1]-2):
            final_form[index,index_2]=position[index,iterator]
            final_form[index,index_2+1]=final_form[index,index_2]+1
            iterator=iterator+1
            index_2=index_2+2
        index=index+1
        index_2=0
    return final_form  
  
def primaryresidualforcevector(loading,connectivity):
    F=np.zeros((20,1))
    mapping=Node_mapping(connectivity)
    index_1=0
    index_2=0
    found=False
    for i in range(loading.shape[0]):
        node=loading[i,0]
        for j_1 in range(connectivity.shape[0]):
            for j_2 in range(connectivity.shape[1]):
                if node==connectivity[j_1,j_2]:
                    index_1=j_1
                    index_2=j_2*2
                    found=True
                    break
            if found:
                found=False
                break
        if loading[i,1]==1:
            F[int(mapping[index_1,index_2]),0]=loading[i,2]
        else:
            F[int(mapping[index_1,index_2+1]),0]=loading[i,2]
    return F 

def reducedresidualloadvector(loading,K,connectivity,boundary):
    loadvector=primaryresidualforcevector(loading, connectivity)
    node=nodehavingdisplacementboundarycondition(boundary, K, connectivity)
    reducedload=np.zeros(loadvector.shape[0]-node.shape[0])
    tracker=0
    unwanted=False
    for i in range(loadvector.shape[0]):
        for j in range(node.shape[0]):
            if i==node[j]:
                unwanted=True
                break
        if unwanted:
            unwanted=False
            continue
        else:
            reducedload[tracker]=loadvector[i]            
            tracker=tracker+1   
    return reducedload

def nodaldisplacement(boundary,K,connectivity,r_dis):
    node=nodehavingdisplacementboundarycondition(boundary, K, connectivity)
    dis=np.zeros(K.shape[0])
    exist=False
    tracker=0
    for i in range(dis.shape[0]):
        for j in range(node.shape[0]):
            if i==node[j]:
                exist=True
                break
        if exist:
            exist=False
            continue
        else:
            dis[i]=r_dis[tracker]
            tracker=tracker+1
    return dis

def nodehavingdisplacementboundarycondition(boundary,K,connectivity):
    node=np.zeros(boundary.shape[0])
    mapping=Node_mapping(connectivity)
    index_1=0
    index_2=0
    found=False
    for i in range(node.shape[0]):
        n=boundary[i,0]
        for j_1 in range(connectivity.shape[0]):
            for j_2 in range(connectivity.shape[1]):
                if n==connectivity[j_1,j_2]:
                    found=True
                    index_1=j_1
                    if n==1:
                        if boundary[i,1]==1:
                            index_2=j_2
                        else:
                            index_2=j_2+1
                    else:
                        if boundary[i,1]==1:
                            index_2=j_2*2
                        else:
                            index_2=j_2*2+1
                    break
            if found:
                found=False
                break
        node[i]=int(mapping[index_1,index_2])
    return node    
                
def ExisitingNode(connectivity,previous_element,current_node):
    row_position=0
    col_position=0
    i=0
    while i<previous_element:
        for j in range(connectivity.shape[1]):
            if connectivity[i,j]==current_node:
                row_position=i
                col_position=j
        i=i+1
    return row_position,col_position

def Chaine_rule(N,J):
    holder=sp.Matrix([[sp.diff(N,n)],[sp.diff(N,e)]])
    J=J.inv()
    solve=J.inv()*holder
    return solve

def switch_case(value,K_1,K_2,K_3,K_4,K_5):
    if value==0:
        return K_1
    if value==1:
        return K_2
    if value==2:
        return K_3
    if value==3:
        return K_4
    if value==4:
        return K_5
   
def StressCalculation(E,B,element,connectivity,nodal_displacement,a,b):
    D=np.zeros((8,1));
    node=Node_mapping(connectivity)[element]
    for i in range(D.shape[0]):
        D[i]=nodal_displacement[int(node[i])]
    stress=E*B*D
    stress=stress.subs({n:a,e:b})
    return stress[0],stress[1]


def reactionforce(k,d):
    force=np.zeros(k.shape[0])
    for i in range(force.shape[0]):
        force[i]=np.dot(k[i,:],d)
    return force
                                   
def NodeElement(inputfile):
    Node=np.zeros((10,2));
    keypoint="null";
    i=0;
    file_path=inputfile;
    with open(file_path,'r')as file:
        for line in file:
            if line=="\n" and keypoint=="Node":
                file.close()
                break
            if keypoint=="Node":
                   spliter=line.split(', ')
                   Node[i,0]=spliter[1]
                   Node[i,1]=spliter[2]
                   i=i+1;
            if "Node" in line:
               keypoint="Node"
    return Node

def ElementConnectivity(inputfile):
    Connectivity=np.zeros((5,4));
    keypoint="null";
    i=0;
    file_path=inputfile;
    with open(file_path,'r')as file:
        for line in file:
            if line=="\n" and keypoint=="Element":
                file.close()
                break
            if keypoint=="Element":
                   spliter=line.split(', ')
                   Connectivity[i,0]=spliter[1]
                   Connectivity[i,1]=spliter[2]
                   Connectivity[i,2]=spliter[3]
                   Connectivity[i,3]=spliter[4]
                   i=i+1;
            if "Element" in line:
               keypoint="Element"
    return Connectivity

def MaterialProperties(inputfile):
    properties=np.zeros(2);
    keypoint="null";
    i=0;
    file_path=inputfile;
    with open(file_path,'r')as file:
        for line in file:
            if line=="\n" and keypoint=="Elastic":
                file.close()
                break
            if keypoint=="Elastic":
                   spliter=line.split(', ')
                   properties[0]=spliter[0]
                   properties[1]=spliter[1]
                   i=i+1;
            if "Elastic" in line:
               keypoint="Elastic"
    return properties

def Load(inputfile):
    loading=np.zeros((5,3))
    keypoint="null";
    i=0;
    file_path=inputfile;
    with open(file_path,'r')as file:
        for line in file:
            if line=="\n" and keypoint=="Dload":
                file.close()
                break
            if keypoint=="Dload":
                   spliter=line.split(', ')
                   loading[i,0]=spliter[0]
                   loading[i,1]=spliter[1]
                   loading[i,2]=spliter[2]
                   i=i+1;
            if "Dload" in line:
               keypoint="Dload"
    return loading

def Boundarycondition(inputfile):
    boundary=np.zeros((7,3))
    keypoint="null";
    i=0;
    file_path=inputfile;
    with open(file_path,'r')as file:
        for line in file:
            if line=="\n" and keypoint=="Boundary":
                file.close()
                break
            if keypoint=="Boundary":
                   spliter=line.split(', ')
                   boundary[i,0]=spliter[0]
                   boundary[i,1]=spliter[1]
                   boundary[i,2]=spliter[2]
                   i=i+1;
            if "Boundary" in line:
               keypoint="Boundary"
    return boundary

#Ouptut result
fea(inputfile)
