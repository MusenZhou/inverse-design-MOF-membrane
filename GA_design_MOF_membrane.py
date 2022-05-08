from pathlib import Path
import subprocess
import os
import pickle
import pormake as pm
import numpy as np
import pygad


# this fxn only builds mof that has correct setting which means the number of node and linkers
# requested in the topology file is satisfied
# for node this fxn automatically assume that node2 would be NA if there is one node input
# if the number of linker in the topo is smaller than 3 linkers, this fxn automatically assumes
# using the first several linkers
def build_MOF(mof_filename, topo, node1, node2, linker1, linker2, linker3):
    global builder
    #
    # topo.describe()
    # node array assign
    N_node = 2
    if node2 == 'NA':
        N_node -= 1
    if node1 == 'NA':
        N_node -= 1
    if topo.n_node_types != N_node:
        print('number of node is NOT compatbile with given node condition')
        return
    if N_node == 2:
        node_bbs = {0: node1, 1: node2,}
    elif N_node == 1:
        node_bbs = {0: node1,}
    # build linker library
    linker_typename_dict = {}
    N_linker_type = 0
    for loop_index1 in range(len(topo.edge_indices)):
        signal_same = 0
        for loop_index2 in range(len(linker_typename_dict)):
            if np.array_equal(linker_typename_dict[loop_index2], topo.get_edge_type(topo.edge_indices[loop_index1])):
                signal_same = 1
                break
        if signal_same == 0:
            linker_typename_dict.update({N_linker_type: topo.get_edge_type(topo.edge_indices[loop_index1])})
            N_linker_type += 1
    # setup linker array
    edge_bbs = {}
    if N_linker_type == 1:
        if linker1 != 'NA':
            edge_bbs.update({(linker_typename_dict[0][0], linker_typename_dict[0][1]): linker1})
    elif N_linker_type == 2:
        if linker1 != 'NA':
            edge_bbs.update({(linker_typename_dict[0][0], linker_typename_dict[0][1]): linker1})
        if linker2 != 'NA':
            edge_bbs.update({(linker_typename_dict[1][0], linker_typename_dict[1][1]): linker2})
    elif N_linker_type == 3:
        if linker1 != 'NA':
            edge_bbs.update({(linker_typename_dict[0][0], linker_typename_dict[0][1]): linker1})
        if linker2 != 'NA':
            edge_bbs.update({(linker_typename_dict[1][0], linker_typename_dict[1][1]): linker2})
        if linker3 != 'NA':
            edge_bbs.update({(linker_typename_dict[2][0], linker_typename_dict[2][1]): linker3})

    # sometimes the constrcution would still fail after all the error catch before-hand
    # escape the construction failure
    try: 
        MOF = builder.build_by_type(topology=topo, node_bbs=node_bbs, edge_bbs=edge_bbs)
        cell_a = MOF.atoms.cell[0,0]
        cell_b = MOF.atoms.cell[1,1]
        cell_c = MOF.atoms.cell[2,2]
        if ((cell_a<50) and (cell_b<50) and (cell_c<50)):
            MOF.write_cif(mof_filename)
            
    except:
        print('construction not working')






def K_gpu_calculation(cif_filename, path_cif2input, path_input_file, \
    path_gpu_K, path_detailed_result):

    global path_ff_table
    command = [path_cif2input, cif_filename, path_ff_table, path_input_file]
    # print(command)
    cif2input = subprocess.Popen(command)
    cif2input.wait()
    # print('A_done')

    command = [path_gpu_K, path_input_file, path_detailed_result]
    # print(command)
    gpu_cal_K = subprocess.Popen(command)
    gpu_cal_K.wait()
    # print('B_done')





def D_gpu_calculation(cif_file_path, filename, components):

    global path_exe_D_cif2input
    global path_ff_table
    global detailed_result_path
    global path_exe_gpu_string_method
    global path_exe_cal_D
    global path_exe_result_cal_D
    # run a initial string method over all directions for all the components considered
    # set up input file so we dont need to redo these all the time
    for i_component in range(len(components)):
        for i_direction in range(1,4,1):
            command = [os.path.join(path_exe_D_cif2input+'_'+components[i_component]+'_'+str(i_direction)), \
            cif_file_path, path_ff_table, \
            os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_'+str(i_direction)+'.input')]
            # print(command)
            cif2input = subprocess.Popen(command)
            cif2input.wait()
    # carry the initial run
    for i_direction in range(1,4,1):
        signal_cal = 0
        for i_component in range(len(components)):
            if not os.path.exists(os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_final_string_result_'+str(i_direction)+'.dat')):
                signal_cal = 1
                print('need to run string method')
                break

        if signal_cal == 1:
            for i_component in range(len(components)):
                command = [path_exe_gpu_string_method, \
                os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_'+str(i_direction)+'.input'), \
                os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_string_result_'+str(i_direction)+'.dat')]
                # print(command)
                gpu_cal_D = subprocess.Popen(command)
                gpu_cal_D.wait()
            D_ini = np.zeros(len(components))
            for i_component in range(len(components)):
                command = [path_exe_cal_D, \
                os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_'+str(i_direction)+'.input'), \
                os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_string_result_'+str(i_direction)+'.dat'), \
                'D_'+components[i_component]+'.dat']
                cal_D = subprocess.Popen(command)
                cal_D.wait()
                with open('D_'+components[i_component]+'.dat', 'r') as file_read:
                    D_ini[i_component] = np.asarray(file_read.readline().split())

            if (np.max(D_ini)>1e-30):
                N_same_ranking = 0
                for i_max_repeat_overheard in range(1,6,1):
                    # calculate the diffusion coefficient of initial string result
                    D_save = np.zeros(len(components))
                    for i_component in range(len(components)):
                        command = [path_exe_cal_D, \
                        os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_'+str(i_direction)+'.input'), \
                        os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_string_result_'+str(i_direction)+'.dat'), \
                        'D_'+components[i_component]+'.dat']
                        cal_D = subprocess.Popen(command)
                        cal_D.wait()
                        with open('D_'+components[i_component]+'.dat', 'r') as file_read:
                            D_save[i_component] = np.asarray(file_read.readline().split())
                    # ranking
                    rank_D = np.argsort(D_save)
                    # convert the ranking to descending
                    rank_D = rank_D[::-1]
                    print('initial D: %.5e %.5e' % (D_save[0], D_save[1]))
                    print('best rank index: %d' % (rank_D[0]))

                    #rerun the string method for the slower diffusion coefficient candidate
                    for i_rerun in range(1, len(rank_D)):
                        D_compare = np.zeros(2)
                        command = [path_exe_gpu_string_method, \
                        os.path.join(detailed_result_path, filename+'_'+components[rank_D[i_rerun]]+'_'+str(i_direction)+'.input'), \
                        os.path.join(detailed_result_path, filename+'_'+components[rank_D[0]]+'_string_result_'+str(i_direction)+'.dat'), \
                        os.path.join(detailed_result_path, filename+'_'+components[rank_D[i_rerun]]+'_new'+'_string_result_'+str(i_direction)+'.dat')]
                        # print(command)
                        gpu_cal_D_rerun = subprocess.Popen(command)
                        gpu_cal_D_rerun.wait()
                        command = [path_exe_cal_D, \
                        os.path.join(detailed_result_path, filename+'_'+components[rank_D[i_rerun]]+'_'+str(i_direction)+'.input'), \
                        os.path.join(detailed_result_path, filename+'_'+components[rank_D[i_rerun]]+'_string_result_'+str(i_direction)+'.dat'), \
                        'D_'+components[rank_D[i_rerun]]+'.dat']
                        cal_D = subprocess.Popen(command)
                        cal_D.wait()
                        with open('D_'+components[rank_D[i_rerun]]+'.dat', 'r') as file_read:
                            D_compare[0] = np.asarray(file_read.readline().split())
                        command = [path_exe_cal_D, \
                        os.path.join(detailed_result_path, filename+'_'+components[rank_D[i_rerun]]+'_'+str(i_direction)+'.input'), \
                        os.path.join(detailed_result_path, filename+'_'+components[rank_D[i_rerun]]+'_new'+'_string_result_'+str(i_direction)+'.dat'), \
                        'D_'+components[rank_D[i_rerun]]+'.dat']
                        cal_D = subprocess.Popen(command)
                        cal_D.wait()
                        with open('D_'+components[rank_D[i_rerun]]+'.dat', 'r') as file_read:
                            D_compare[1] = np.asarray(file_read.readline().split())
                        print('old D: %.5e new D: %.5e' % (D_compare[0], D_compare[1]))
                        if (D_compare[1]>D_compare[0]):
                            command = ['mv', os.path.join(detailed_result_path, filename+'_'+components[rank_D[i_rerun]]+'_new'+'_string_result_'+str(i_direction)+'.dat'), \
                            os.path.join(detailed_result_path, filename+'_'+components[rank_D[i_rerun]]+'_string_result_'+str(i_direction)+'.dat')]
                            mv_string_result = subprocess.Popen(command)
                            mv_string_result.wait()
                            print('update')
                        else:
                            os.remove(os.path.join(detailed_result_path, filename+'_'+components[rank_D[i_rerun]]+'_new'+'_string_result_'+str(i_direction)+'.dat'))
                            print('not update')
                            # command = ['rm', ]

                    D_new = np.zeros(len(components))
                    for i_component in range(len(components)):
                        command = [path_exe_cal_D, \
                        os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_'+str(i_direction)+'.input'), \
                        os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_string_result_'+str(i_direction)+'.dat'), \
                        'D_'+components[i_component]+'.dat']
                        cal_D = subprocess.Popen(command)
                        cal_D.wait()
                        with open('D_'+components[i_component]+'.dat', 'r') as file_read:
                            D_new[i_component] = np.asarray(file_read.readline().split())
                    # ranking
                    rank_new_D = np.argsort(D_new)
                    # convert the ranking to descending
                    rank_new_D = rank_new_D[::-1]
                    print('update D: %.5e %.5e' % (D_new[0], D_new[1]))
                    print('best rank index: %d' % (rank_new_D[0]))

                    if np.array_equal(rank_D, rank_new_D):
                        N_same_ranking += 1
                        print('good same')
                    else:
                        N_same_ranking = 0
                        print('bad different')
                    if N_same_ranking == 2:
                        print('escape good')
                        for i_component in range(len(components)):
                            command = ['mv', os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_string_result_'+str(i_direction)+'.dat'), \
                            os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_final_string_result_'+str(i_direction)+'.dat')]
                            # print(command)
                            mv_string_result = subprocess.Popen(command)
                            mv_string_result.wait()
                        break
            else:
                for i_component in range(len(components)):
                    command = ['mv', os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_string_result_'+str(i_direction)+'.dat'), \
                    os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_final_string_result_'+str(i_direction)+'.dat')]
                    # print(command)
                    mv_string_result = subprocess.Popen(command)
                    mv_string_result.wait()

        # else:
        #     print('no need for calculation')

    D_summary = np.zeros(len(components))
    for i_component in range(len(components)):
        for i_direction in range(1,4,1):
            command = [path_exe_result_cal_D, \
            os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_'+str(i_direction)+'.input'), \
            os.path.join(detailed_result_path, filename+'_'+components[i_component]+'_final_string_result_'+str(i_direction)+'.dat'), \
            'D_'+components[i_component]+'.dat']
            cal_D = subprocess.Popen(command)
            cal_D.wait()
            D_read = np.zeros(1)
            with open('D_'+components[i_component]+'.dat', 'r') as file_read:
                D_read[0] = np.asarray(file_read.readline().split())
                # print('D_read: %.5e' % (D_read[0]))
            D_summary[i_component] += D_read[0]
        D_summary[i_component] = 1.0*(D_summary[i_component])/3.0
    # print(D_summary)


    return D_summary

        



















# the solution as input would be integer that refers back to the spcific topology, node and linker
def MOF_evaluate(solution):

    global pormake_topo_list
    global pormake_node_list
    global pormake_linker_list

    global result_path
    global built_cif_path
    global detailed_result_path

    global pormake_database

    global locator

    global path_exe_ethane_cif2input
    global path_exe_ethene_cif2input
    global path_exe_gpu_Vext_poly_Henry



    select_topo_name = pormake_topo_list[solution[0]]
    select_topo = pormake_database.get_topo(pormake_topo_list[solution[0]])
    select_node1_name = pormake_node_list[solution[1]]
    select_node1 = pormake_database.get_bb(pormake_node_list[solution[1]])
    if solution[2]<0:
        select_node2 = 'NA'
        select_node2_name = 'NA'
    else:
        select_node2 = pormake_database.get_bb(pormake_node_list[solution[2]])
        select_node2_name = pormake_node_list[solution[2]]


    if solution[3]<0:
        select_linker1 = 'NA'
        select_linker1_name = 'NA'
    else:
        select_linker1 = pormake_database.get_bb(pormake_linker_list[solution[3]])
        select_linker1_name = pormake_linker_list[solution[3]]


    if solution[4]<0:
        select_linker2 = 'NA'
        select_linker2_name = 'NA'
    else:
        select_linker2 = pormake_database.get_bb(pormake_linker_list[solution[4]])
        select_linker2_name = pormake_linker_list[solution[3]]


    if solution[5]<0:
        select_linker3 = 'NA'
        select_linker3_name = 'NA'
    else:
        select_linker3 = pormake_database.get_bb(pormake_linker_list[solution[5]])
        select_linker3_name = pormake_linker_list[solution[3]]

    # print(select_topo)
    # print(pormake_topo_list[solution[0]])
    filename = os.path.join(select_topo_name+'_'\
        +select_node1_name+'_'+select_node2_name+'_'\
        +select_linker1_name+'_'+select_linker2_name+'_'+select_linker3_name)

    cif_file_path = os.path.join(built_cif_path, filename+'.cif')
    result_file_path = os.path.join(result_path, filename+'.dat')

    if os.path.isfile(result_file_path):
        ## read performance
        ## result are saved in the following format:
        ## K_ethane K_ethene D_ethane D_ethene
        ## K is in the dimensionless unit
        ## D is in the unit of m2/s but is already averaged over three dimensions
        print('file existed!!!!')
        K_ethane = np.zeros(1)
        K_ethene = np.zeros(1)
        D_ethane = np.zeros(1)
        D_ethene = np.zeros(1)
        with open(result_file_path, 'r') as file_read:
            result = np.asarray(file_read.readline().split())

        K_ethane[0] = result[0]
        K_ethene[0] = result[1]
        D_ethane[0] = result[2]
        D_ethene[0] = result[3]
        print('%.5e %.5e %.5e %.5e' %(K_ethane[0], K_ethene[0], D_ethane[0], D_ethene[0]))
    else:
        print('evaluating')
        # print(solution)
        # build mof_file
        signal_mof_build = 1

        if solution[2]<0:
            N_assigned_node = 1
        else:
            N_assigned_node = 2



        if select_topo.n_node_types != N_assigned_node:
            signal_mof_build = 0

        if ( (select_topo.n_edge_types<3) and (solution[5]>=0) ):
            signal_mof_build = 0
        elif ( (select_topo.n_edge_types<2) and (solution[4]>=0) ):
            signal_mof_build = 0
        elif ( (select_topo.n_edge_types<1) and (solution[3]>=0) ):
            signal_mof_build = 0

        # check whether the cn on node is enough for the given topology
        if (N_assigned_node == 2) and (select_topo.n_node_types==2):
            if ( (select_node1.n_connection_points < select_topo.unique_cn[0]) or \
                        (select_node2.n_connection_points < select_topo.unique_cn[0]) ):
                signal_mof_build = 0
            # elif ((select_topo.unique_cn[0]%2==1) and (select_node1.n_connection_points%2==0)): 
            #     signal_mof_build = 0
            # elif ((select_topo.unique_cn[1]%2==1) and (select_node2.n_connection_points%2==0)): 
            #     signal_mof_build = 0
            # elif ((select_topo.unique_cn[0]%2==0) and (select_node1.n_connection_points%2==1)): 
            #     signal_mof_build = 0
            # elif ((select_topo.unique_cn[1]%2==0) and (select_node2.n_connection_points%2==1)): 
            #     signal_mof_build = 0
            else:
                try:
                    locator_score_1 = locator.calculate_rmsd(select_topo.unique_local_structures[0], select_node1)
                except:
                    signal_mof_build = 0

                try:
                    locator_score_2 = locator.calculate_rmsd(select_topo.unique_local_structures[1], select_node2)
                except:
                    signal_mof_build = 0
                    
                if signal_mof_build == 1:
                    if ( (locator_score_1 > 0.3) or (locator_score_2 > 0.3) ):
                        signal_mof_build = 0
        elif (N_assigned_node == 1) and (select_topo.n_node_types==1):
            if (select_node1.n_connection_points < select_topo.unique_cn[0]):
                signal_mof_build = 0
            # elif ((select_topo.unique_cn[0]%2==1) and (select_node1.n_connection_points%2==0)): 
            #     signal_mof_build = 0
            # elif ((select_topo.unique_cn[0]%2==0) and (select_node1.n_connection_points%2==1)): 
            #     signal_mof_build = 0
            else:
                locator_score_1 = locator.calculate_rmsd(select_topo.unique_local_structures[0], select_node1)
                if locator_score_1 > 0.3:
                    signal_mof_build = 0
        else:
            signal_mof_build = 0
        
        if signal_mof_build == 1:
            #mof cif can be built

            # try to build mof with given structure
            print(solution)
            print('MOF can be built')
            print(cif_file_path)

            build_MOF(cif_file_path, select_topo, select_node1, select_node2, select_linker1, \
                select_linker2, select_linker3)

            if os.path.isfile(cif_file_path):
                # evaluate henry's constant 
                # evaluate henry's constant 
                # evaluate henry's constant 
                # evaluate henry's constant 
                # evaluate henry's constant 
                path_detailed_result = os.path.join(detailed_result_path, filename+'_ethane.dat')
                # print(path_detailed_result)
                path_input_file =  os.path.join(detailed_result_path, filename+'_K_ethane.input')
                K_ethane = np.zeros(1)
                if not os.path.isfile(path_detailed_result):
                    K_gpu_calculation(cif_file_path, path_exe_ethane_cif2input, path_input_file, \
                        path_exe_gpu_Vext_poly_Henry, path_detailed_result)
                    with open(path_detailed_result, 'r') as file_read:
                        read_value = file_read.readline().split()
                    # first value is number of pariwise interaction evaluate
                    # second value the dimensionless henry's constant ###################
                    # thrid value is the time GPU program used to calculate
                    K_ethane[0] = read_value[1]
                else:
                    with open(path_detailed_result, 'r') as file_read:
                        read_value = file_read.readline().split()
                    # first value is number of pariwise interaction evaluate
                    # second value the dimensionless henry's constant ###################
                    # thrid value is the time GPU program used to calculate
                    K_ethane[0] = read_value[1]


                path_detailed_result = os.path.join(detailed_result_path, filename+'_ethene.dat')
                # print(path_detailed_result)
                path_input_file =  os.path.join(detailed_result_path, filename+'_K_ethene.input')
                K_ethene = np.zeros(1)
                if not os.path.isfile(path_detailed_result):
                    K_gpu_calculation(cif_file_path, path_exe_ethene_cif2input, path_input_file, \
                        path_exe_gpu_Vext_poly_Henry, path_detailed_result)
                    with open(path_detailed_result, 'r') as file_read:
                        read_value = file_read.readline().split()
                    # first value is number of pariwise interaction evaluate
                    # second value the dimensionless henry's constant ###################
                    # thrid value is the time GPU program used to calculate
                    # K_ethene[0] = np.asarray(read_value[1], dtype=np.float32)
                    K_ethene[0] = read_value[1]
                else:
                    with open(path_detailed_result, 'r') as file_read:
                        read_value = file_read.readline().split()
                    # first value is number of pariwise interaction evaluate
                    # second value the dimensionless henry's constant ###################
                    # thrid value is the time GPU program used to calculate
                    # K_ethene[0] = np.asarray(read_value[1], dtype=np.float32)
                    K_ethene[0] = read_value[1]
                # print()
                print('K_value: %.5e %.5e' % (K_ethane[0], K_ethene[0]))
                # print(K_ethane)
                # print(K_ethene)

                # evaluate diffusion coefficient
                D_ethane = np.zeros(1)
                D_ethene = np.zeros(1)
                components = ['ethane', 'ethene']
                [D_ethane[0], D_ethene[0]] = D_gpu_calculation(cif_file_path, filename, components)
                print('D_value: %.5e %.5e' % (D_ethane[0], D_ethene[0]))

                ## result are saved in the following format:
                ## K_ethane K_ethene D_ethane D_ethene
                ## K is in the dimensionless unit
                ## D is in the unit of m2/s but is already averaged over three dimensions
                with open(result_file_path, 'w') as file_write:
                    file_write.write('%.5e %.5e %.5e %.5e\n' %(K_ethane[0], K_ethene[0], D_ethane[0], D_ethene[0]))
            else:
                # for somewhat reason pormake cannot build this mof structure file
                print(solution)
                print('MOF cannot be built')
                with open(result_file_path, 'w') as file_write:
                    file_write.write('-999 -999 -999 -999\n')
                K_ethane = np.zeros(1)
                K_ethene = np.zeros(1)
                D_ethane = np.zeros(1)
                D_ethene = np.zeros(1)
                K_ethane[0] = -999
                K_ethene[0] = -999
                D_ethane[0] = -999
                D_ethene[0] = -999

        else:
            # mof cif cannot be built
            print(solution)
            print('MOF cannot be built')
            with open(result_file_path, 'w') as file_write:
                file_write.write('-999 -999 -999 -999\n')
            K_ethane = np.zeros(1)
            K_ethene = np.zeros(1)
            D_ethane = np.zeros(1)
            D_ethene = np.zeros(1)
            K_ethane[0] = -999
            K_ethene[0] = -999
            D_ethane[0] = -999
            D_ethene[0] = -999


    return K_ethane[0], K_ethene[0], D_ethane[0], D_ethene[0]





# the solution as input would be integer that refers back to the spcific topology, node and linker
def MOF_performance_extra(solution):

    global pormake_topo_list
    global pormake_node_list
    global pormake_linker_list

    global result_path
    global built_cif_path
    global detailed_result_path

    global pormake_database

    global locator

    global path_exe_ethane_cif2input
    global path_exe_ethene_cif2input
    global path_exe_gpu_Vext_poly_Henry



    select_topo_name = pormake_topo_list[solution[0]]
    select_topo = pormake_database.get_topo(pormake_topo_list[solution[0]])
    select_node1_name = pormake_node_list[solution[1]]
    select_node1 = pormake_database.get_bb(pormake_node_list[solution[1]])
    if solution[2]<0:
        select_node2 = 'NA'
        select_node2_name = 'NA'
    else:
        select_node2 = pormake_database.get_bb(pormake_node_list[solution[2]])
        select_node2_name = pormake_node_list[solution[2]]


    if solution[3]<0:
        select_linker1 = 'NA'
        select_linker1_name = 'NA'
    else:
        select_linker1 = pormake_database.get_bb(pormake_linker_list[solution[3]])
        select_linker1_name = pormake_linker_list[solution[3]]


    if solution[4]<0:
        select_linker2 = 'NA'
        select_linker2_name = 'NA'
    else:
        select_linker2 = pormake_database.get_bb(pormake_linker_list[solution[4]])
        select_linker2_name = pormake_linker_list[solution[3]]


    if solution[5]<0:
        select_linker3 = 'NA'
        select_linker3_name = 'NA'
    else:
        select_linker3 = pormake_database.get_bb(pormake_linker_list[solution[5]])
        select_linker3_name = pormake_linker_list[solution[3]]

    # print(select_topo)
    # print(pormake_topo_list[solution[0]])
    filename = os.path.join(select_topo_name+'_'\
        +select_node1_name+'_'+select_node2_name+'_'\
        +select_linker1_name+'_'+select_linker2_name+'_'+select_linker3_name)

    cif_file_path = os.path.join(built_cif_path, filename+'.cif')
    result_file_path = os.path.join(result_path, filename+'.dat')

    if os.path.isfile(result_file_path):
        ## read performance
        ## result are saved in the following format:
        ## K_ethane K_ethene D_ethane D_ethene
        ## K is in the dimensionless unit
        ## D is in the unit of m2/s but is already averaged over three dimensions
        print('file existed!!!!')
        K_ethane = np.zeros(1)
        K_ethene = np.zeros(1)
        D_ethane = np.zeros(1)
        D_ethene = np.zeros(1)
        with open(result_file_path, 'r') as file_read:
            result = np.asarray(file_read.readline().split())

        K_ethane[0] = result[0]
        K_ethene[0] = result[1]
        D_ethane[0] = result[2]
        D_ethene[0] = result[3]

        if (np.isnan(D_ethane) or np.isnan(D_ethene)):
            D_ethane[0] = 0
            D_ethene[0] = 0
        
        print('%.5e %.5e %.5e %.5e' %(K_ethane[0], K_ethene[0], D_ethane[0], D_ethene[0]))
    else:
        print('!!!!!!!!!error!!!!!!!!!!!!no result file can be found')
        print('!!!!!!!!!error!!!!!!!!!!!!no result file can be found')
        print('!!!!!!!!!error!!!!!!!!!!!!no result file can be found')
        print('!!!!!!!!!error!!!!!!!!!!!!no result file can be found')
        print('!!!!!!!!!error!!!!!!!!!!!!no result file can be found')
        print('!!!!!!!!!error!!!!!!!!!!!!no result file can be found')
        print('!!!!!!!!!error!!!!!!!!!!!!no result file can be found')
        print('!!!!!!!!!error!!!!!!!!!!!!no result file can be found')


    return K_ethane[0], K_ethene[0], D_ethane[0], D_ethene[0]

















def MOF_fitness(solution, solution_idx):

    # print(solution)
    K_ethane, K_ethene, D_ethane, D_ethene = MOF_evaluate(solution)

    if (np.isnan(D_ethane) or np.isnan(D_ethene)):
        D_ethane = 0
        D_ethene = 0

    ## figure out something for fitness score
    if K_ethane < 0:
        fitness = 1
    elif ((D_ethene<1e-30) and (D_ethane<1e-30)):
        # not permeable
        fitness = 1
    else:
        mem_selectivity = 1.0*K_ethene*D_ethene/(K_ethane*D_ethane)
        permeability_barrer = (1.0*(K_ethene/(8.314*300))*D_ethene)/3.35*1e16
        # fitness  = 0.5*np.exp(np.log10(mem_selectivity-5)) + 0.5*np.exp(1.0*np.log10(permeability_barrer-95)/3.2)
        # fitness  = np.log10(mem_selectivity-5)
        fitness = 0
        if (mem_selectivity>5):
            # fitness += 0.5*np.log10(mem_selectivity-5)
            # fitness += 0.5*(np.exp(mem_selectivity-5)+1)
            fitness += 0.5*(np.power((mem_selectivity-5),2)+2)
        else:
            fitness += 0.5*(np.exp(mem_selectivity-5)+1)

        if (permeability_barrer>95):
            # fitness += 0.5*(np.log10(permeability_barrer-95)/3.2)
            # fitness += 0.5*(np.log10(permeability_barrer-95)/20+1)
            fitness += 0.5*(np.log10(permeability_barrer-95)/100+1)
        else:
            fitness += 0.5*(np.exp(permeability_barrer-95)+1)
        print('selectivity: %.5e permeability_barrer: %.5e' %(mem_selectivity, permeability_barrer))
        print('fitness %.5e' %(fitness))

    return fitness





def fitness_func(solution, solution_idx):
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    output = np.sum(solution*function_inputs)
    # The value 0.000001 is used to avoid the Inf value when the denominator numpy.abs(output - desired_output) is 0.0.
    fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
    return fitness





path_save_pormake_topo_list = './full_pormake_topo_list.p'
path_save_pormake_node_list = './full_pormake_node_list.p'
path_save_pormake_linker_list = './full_pormake_linker_list.p'

path_exe_ethane_cif2input = './utility/pormake_cif2input_ethane'
path_exe_ethene_cif2input = './utility/pormake_cif2input_ethene'
path_exe_gpu_Vext_poly_Henry = './utility/Vext_poly_Henry'
path_ff_table = './utility/data_ff_UFF'
path_exe_D_cif2input = './utility/D_cif2input'
path_exe_gpu_string_method = './utility/GPU_string_method_cal'
path_exe_cal_D = './utility/cal_D'
path_exe_result_cal_D = './utility/result_cal_D'





with open(path_save_pormake_topo_list, 'rb') as f_read:
    pormake_topo_list = pickle.load(f_read)

with open(path_save_pormake_node_list, 'rb') as f_read:
    pormake_node_list = pickle.load(f_read)

with open(path_save_pormake_linker_list, 'rb') as f_read:
    pormake_linker_list = pickle.load(f_read)

result_path = './archive_result'
built_cif_path = './cif_pormake_build'
detailed_result_path = './detailed_result'


if not os.path.isdir(result_path):
    os.mkdir(result_path)


if not os.path.isdir(built_cif_path):
    os.mkdir(built_cif_path)


if not os.path.isdir(detailed_result_path):
    os.mkdir(detailed_result_path)


pormake_database = pm.Database()
builder = pm.Builder()
locator = pm.Locator()


############## Genetic algorithm parameters setting ##############
############## Genetic algorithm parameters setting ##############
############## Genetic algorithm parameters setting ##############

fitness_function = MOF_fitness


# number of iterations
num_generations = 1
# the actual number of iteration in this implementation
N_iteration = 3
# number of solutions to be selected as parents
num_parents_mating = 10
# number of solutions in each iterations
sol_per_pop = 2000
# number of variables in each solution
num_genes = 6
# ssss - steady-state seletion; rws - roulette wheel selection
# sus - stochastic universal selection, random - randome selection
# tournament - tournament selection
parent_selection_type = "sus"
# keep one parent from last iteration to next
keep_parents = 2


# ???
crossover_type = "single_point"
# ????
mutation_type = "random"
mutation_percent_genes = 30








############## Genetic algorithm parameters setting ##############
############## Genetic algorithm parameters setting ##############
############## Genetic algorithm parameters setting ##############
############## Genetic algorithm parameters setting ##############
############## Genetic algorithm parameters setting ##############


topo_low = 0
topo_high = len(pormake_topo_list)
node1_low = 0
node1_high = len(pormake_node_list)
node2_low = -1
node2_high = len(pormake_node_list)

node2_database = np.zeros(2*len(pormake_node_list))
for i_number in range(len(pormake_node_list)):
    node2_database[i_number*2] = i_number
    node2_database[i_number*2+1] = -1
node2_database = np.int64(node2_database)

linker1_database = np.zeros(2*len(pormake_linker_list))
for i_number in range(len(pormake_linker_list)):
    linker1_database[i_number*2] = i_number
    linker1_database[i_number*2+1] = -1
linker1_database = np.int64(linker1_database)
linker2_database = np.int64(linker1_database)
linker3_database = np.int64(linker1_database)

gene_space = [np.arange(topo_low, topo_high, 1), np.arange(node1_low, node1_high, 1), node2_database, \
linker1_database, linker2_database, linker3_database]


np.random.seed(0)
topo_ini = np.random.randint(topo_low, topo_high, sol_per_pop)
np.random.seed(1)
node1_ini = np.random.randint(node1_low, node1_high, sol_per_pop)
np.random.seed(2)
node2_ini = node2_database[np.random.choice(len(node2_database), size=sol_per_pop)]
np.random.seed(3)
linker1_ini = linker1_database[np.random.choice(len(linker1_database), size=sol_per_pop)]
np.random.seed(4)
linker2_ini = linker1_database[np.random.choice(len(linker2_database), size=sol_per_pop)]
np.random.seed(5)
linker3_ini = linker1_database[np.random.choice(len(linker3_database), size=sol_per_pop)]
ini_guess = np.vstack([topo_ini, node1_ini, node2_ini, linker1_ini, linker2_ini, linker3_ini] ).transpose()





save_solutions_full = np.zeros(( sol_per_pop, num_genes, (N_iteration+1) ))
top_percent = 20

top_tier_fitness = np.zeros(((N_iteration+1), int(np.floor(sol_per_pop*top_percent/100))))
top_tier_solutions = np.zeros(( int(np.floor(sol_per_pop*top_percent/100)),  num_genes, (N_iteration+1) ))
top_tier_result = np.zeros(( int(np.floor(sol_per_pop*top_percent/100)), 4, (N_iteration+1) ))

current_soolutions = ini_guess
current_soolutions.astype(int)

save_solutions_full[:, :, 0] = current_soolutions

ini_fitness = np.zeros((sol_per_pop,))
for index1 in range(sol_per_pop):
    solution = current_soolutions[index1,:]
    ini_fitness[index1] = MOF_fitness(solution, [])

sort_inin_fitness = np.sort(ini_fitness)
sort_inin_fitness = sort_inin_fitness[::-1]
top_tier_fitness[0,:] = sort_inin_fitness[0:int(np.floor(sol_per_pop*top_percent/100))]
sort_rank_ini_fitness = np.argsort(ini_fitness)
sort_rank_ini_fitness = sort_rank_ini_fitness[::-1]
top_tier_solutions[:,:,0] = current_soolutions[sort_rank_ini_fitness[0:int(np.floor(sol_per_pop*top_percent/100))],:]
print(top_tier_solutions[:,:,0])
print(top_tier_fitness)



######## archive the detailed solutions
######## archive the detailed solutions
for index2 in range(int(np.floor(sol_per_pop*top_percent/100))):
    solu_calc = np.int64(top_tier_solutions[index2,:,0])
    # solu_calc.astype(np.int64)
    print(solu_calc)
    temp_K_ethane, temp_K_ethene, temp_D_ethane, temp_D_ethene = MOF_performance_extra(solu_calc)
    top_tier_result[index2,:,0] = [temp_K_ethane, temp_K_ethene, temp_D_ethane, temp_D_ethene]

print(top_tier_result[:,:,0])

np.savez('conclude.npz', top_tier_solutions=top_tier_solutions, top_tier_fitness=top_tier_fitness, \
    top_tier_result=top_tier_result, save_solutions_full=save_solutions_full)




for index1 in range(N_iteration):
    print('generation: %d' %(index1))
    ga_instance = pygad.GA(num_generations=num_generations, 
                           num_parents_mating=num_parents_mating, 
                           fitness_func=MOF_fitness,
                           initial_population=ini_guess, 
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space = gene_space,
                           gene_type=int, 
                           # init_range_low=init_range_low,
                           # init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes="default", 
                           save_solutions=True)
    ga_instance.run()
    save_solutions_full[:, :, index1+1] = ga_instance.solutions[(ga_instance.solutions.shape[0]-sol_per_pop):(ga_instance.solutions.shape[0]),:]
    ini_guess = ga_instance.solutions[(ga_instance.solutions.shape[0]-sol_per_pop):(ga_instance.solutions.shape[0]),:]
    current_soolutions = ga_instance.solutions[(ga_instance.solutions.shape[0]-sol_per_pop):(ga_instance.solutions.shape[0]),:]
    #
    print(current_soolutions)
    fitness_last_generation = np.sort(ga_instance.last_generation_fitness)
    fitness_last_generation = fitness_last_generation[::-1]
    fitness_rank_last_generation = np.argsort(ga_instance.last_generation_fitness)
    fitness_rank_last_generation = fitness_rank_last_generation[::-1]
    top_tier_fitness[index1+1,:] = fitness_last_generation[0:int(np.floor(sol_per_pop*top_percent/100))]
    top_tier_solutions[:,:,(index1+1)] = current_soolutions[fitness_rank_last_generation[0:int(np.floor(sol_per_pop*top_percent/100))],:]
    print(top_tier_solutions[:,:,(index1+1)])
    print(top_tier_fitness[index1+1,:])

    ######## need to archive the detailed solutions
    ######## need to archive the detailed solutions
    ######## need to archive the detailed solutions
    ######## need to archive the detailed solutions
    #
    for index2 in range(int(np.floor(sol_per_pop*top_percent/100))):
        solu_calc = np.int64(top_tier_solutions[index2,:,(index1+1)])
        print(solu_calc)
        temp_K_ethane, temp_K_ethene, temp_D_ethane, temp_D_ethene = MOF_performance_extra(solu_calc)
        # print(top_tier_result[index2,(index1+1)*4:(index1+2)*4])
        # print([temp_K_ethane, temp_K_ethene, temp_D_ethane, temp_D_ethene])
        top_tier_result[index2,:,(index1+1)] = [temp_K_ethane, temp_K_ethene, temp_D_ethane, temp_D_ethene]

    print(top_tier_result[:,:,(index1+1)])
    np.savez('conclude.npz', top_tier_solutions=top_tier_solutions, top_tier_fitness=top_tier_fitness, \
        top_tier_result=top_tier_result, save_solutions_full=save_solutions_full)
















