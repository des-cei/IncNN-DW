from cProfile import label
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcursors
import os
import pickle

# Path to store obtained results
os.chdir("Results/Testing_dictionaries")

with open('Resultados_FL_sr1_all_outlier_free.pkl', 'rb') as handle:
    results = pickle.load(handle)
handle.close()


model_type = ['Top-power', 'Bottom-power', 'Time']
result_type = ["Kernel_type_1", "Kernel_type_2", "Kernel_type_3", "Full_test"]
results_types = ['Mean', 'sdv', 'Time', 'Model_name', 'User']

# Path to store obtained results
os.chdir("../Pareto_all_no_outliers")

# If the user just want to see the Pareto this can be done for multiple variable parameters
# Otherwise just 2 could be display, so some comprobations need to be performced
# Matplotlib configuration
mpl.rcParams['figure.figsize'] = (15, 8)
# Remove top and right frame
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.bottom'] = True

# User help info for possible actions
print("Actions on red dots (Pareto Frontier):")
print("\t(mouse) hover: Display iteration information as an annotation.")
print("\t(mouse) left Click: print iteration information in terminal.")
print("\t(mouse) right Click (on annotation): removes previously generated annotation.")
print("\t(keyboard) 'v': toggles annotation's visibility.\n")

# Variable combination for pareto
mode = 0
name = ["MAPE mean (%)", "MAPE sdv (%)", "MAPE mean (%)", "Time"]

# Iterate over each model type
for model in model_type:
    # Print model type in terminal
    print("\n{}".format(model.replace('_', ' ').capitalize()))

    for rt in result_type:
        print(f"{rt}")
        number_models = len(results[model][rt])
        data = []
        for i in range(number_models):
            Mean = float(results[model][rt][str(i)][results_types[0]]) # MAPE mean
            sdv = float(results[model][rt][str(i)][results_types[1]]) # MAPE sdv
            Time = float(results[model][rt][str(i)][results_types[2]]) # MAPE sdv
            model_name = results[model][rt][str(i)][results_types[3]] # Model name
            user = results[model][rt][str(i)][results_types[4]] # user

            data.append((model_name, Mean, sdv, Mean, Time, i, user))

        for mode in range(len(name)/2):
            # Generate a list of points that will conform the pareto points
            # For each point in data we get its id (iteration), mape and %train
            pareto_points = []
            for i in range(number_models):
                pareto_points.append({"Model name": data[i][0], name[2*mode]: data[i][mode*2+1], name[2*mode+1]: data[i][mode*2+2], "index": data[i][5], "user": data[i][6]})

            print(f"Pareto {name[2*mode]} vs {name[2*mode+1]}, {len(pareto_points)} total of pareto points")

            server_pareto = []
            client1_pareto = []
            client2_pareto = []
            client3_pareto = []
            for point in pareto_points:
                if point["user"] == "Server_side":
                    server_pareto.append(point)
                if point["user"] == "Client_1":
                    client1_pareto.append(point)
                if point["user"] == "Client_2":
                    client2_pareto.append(point)
                if point["user"] == "Client_3":
                    client3_pareto.append(point)

            # Extract pareto frontier x and y values as well as ids for plotting
            server_x_values = [point[name[2*mode]] for point in server_pareto]
            server_y_values = [point[name[2*mode+1]] for point in server_pareto]
            server_names = [point["Model name"] for point in server_pareto]
            server_ids = [point["index"] for point in server_pareto]  # Extract IDs as strings

            # Extract non pareto frontier x and y values for plotting
            client1_x_values = [point[name[2*mode]] for point in client1_pareto]
            client1_y_values = [point[name[2*mode+1]] for point in client1_pareto]
            client1_names = [point["Model name"] for point in client1_pareto]
            client1_ids = [point["index"] for point in client1_pareto]  # Extract IDs as strings

            # Extract non pareto frontier x and y values for plotting
            client2_x_values = [point[name[2*mode]] for point in client2_pareto]
            client2_y_values = [point[name[2*mode+1]] for point in client2_pareto]
            client2_names = [point["Model name"] for point in client2_pareto]
            client2_ids = [point["index"] for point in client2_pareto]  # Extract IDs as strings

            # Extract non pareto frontier x and y values for plotting
            client3_x_values = [point[name[2*mode]] for point in client3_pareto]
            client3_y_values = [point[name[2*mode+1]] for point in client3_pareto]
            client3_names = [point["Model name"] for point in client3_pareto]
            client3_ids = [point["index"] for point in client3_pareto]  # Extract IDs as strings

            # Generate the pareto

            ##########
            ## Plot ##
            ##########

            # Create a figure
            fig, axis = plt.subplots(constrained_layout=True)

            # Select the figure title
            if model == "Top-power":
                plot_title = "Top Power Model " + rt
            elif model == "Bottom-power":
                plot_title = "Bottom Power Model " + rt
            else:
                plot_title = "Time Model " + rt

            # Plot the Pareto frontier
            server_scatter = axis.scatter(server_x_values, server_y_values, label='Server', color='b', edgecolors='black', marker='s', s=20)#s=25)
            client1_scatter = axis.scatter(client1_x_values, client1_y_values, label='Client 1', color='r', edgecolors='black', marker='o')
            client2_scatter = axis.scatter(client2_x_values, client2_y_values, label='Client 2', color='y', edgecolors='black', marker='o')
            client3_scatter = axis.scatter(client3_x_values, client3_y_values, label='Client 3', color='g', edgecolors='black', marker='o')


            #########################################
            ## Local functions for cursor acctions ##
            #########################################

            def get_iteration_info(sel_index):
                "Generate a string with the information of this particular iteration"
                # Get the iteration to which the user is pointing with the cursor
                index = server_ids[sel_index]

                # Get the index of that iteration within the data error and %train lists
                for i in range(number_models):
                    if (data[i][5] == index):
                        actual_index = i

                model_name = data[actual_index][0]
                # Introduce the iteration value in the string
                resulting_string = "Model name: {}\n".format(model_name)

                resulting_string += "----------------------------\n"

                # Concatenate each variable parameter
                i = 0
                for j in range(3):
                    place = model_name.find(',', i)
                    substring = model_name[i:place]
                    resulting_string += substring + "\n"
                    i = place + 2
                substring=model_name[i:]
                resulting_string += substring + "\n"

                # Concatenate training error and % of trained observations
                resulting_string += "----------------------------\n"
                resulting_string += "MAPE mean: {:.3f} %\n".format(data[actual_index][1])
                resulting_string += "MAPE sdv: {:.3f} %\n".format(data[actual_index][2])
                resulting_string += "Time: {:.3f} s\n".format(data[actual_index][4])
                resulting_string += "\nActual index: " + str(actual_index)

                # Return the string
                return resulting_string

            def cursor_annotation(sel):
                """Generate an annotation with iteration information when hovering cursor on pareto-optimal points"""

                # Generate de annotation (calling the function that generates the iteration info)
                sel.annotation.set_text(get_iteration_info(sel.index))
                # Change annotation transparency
                sel.annotation.get_bbox_patch().set(alpha=1)

            def cursor_print(sel):
                """Printing on terminal iteration information when clicking cursor on pareto-optimal points"""

                # Print the iteration information
                print(get_iteration_info(sel.index))
                print("----------------------------\n")
                # Make annotation invisible
                sel.annotation.set(visible=False)

            #############
            ## Cursors ##
            #############

            # Enable hover so when the user hovers the mouse over a point the annotation is generated
            cursor1 = mplcursors.cursor(server_scatter, hover=True)
            cursor1.connect("add", cursor_annotation)

            # Disable hover so when the user hovers the mouse over a point nothing happens
            # But when clicking the info of that iteration is printed
            cursor2 = mplcursors.cursor(server_scatter, hover=False)
            cursor2.connect("add", cursor_print)

            axis.minorticks_on()
            #axes.grid(alpha=0.5, linestyle=':', which="both")
            # Or if you want different settings for the grids:
            axis.grid(which='minor', linestyle=':', alpha=0.2)
            axis.grid(which='major', linestyle=':', alpha=0.5)
            plt.title(plot_title)
            plt.xlabel(name[2*mode], fontsize=20)
            plt.ylabel(name[2*mode+1], fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)

            axis.legend()

            plt.savefig("Pareto_all_no_outliers_" + model + "_" + rt + "_sr1_" + name[2*mode] + "_vs_" + name[2*mode+1] + ".tiff")
            plt.show()
        print("\n")
exit()


