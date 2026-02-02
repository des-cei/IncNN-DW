from cProfile import label
from pprint import pprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcursors
import os
import pickle

# Path to store obtained results
os.chdir("Results/Testing_dictionaries")

with open('Resultados_FL_sr1_server.pkl', 'rb') as handle:
    results = pickle.load(handle)
handle.close()


model_type = ['Top-power', 'Bottom-power', 'Time']
result_type = ["Kernel_type_1", "Kernel_type_2", "Kernel_type_3", "Full_test"]
results_types = ['Mean', 'sdv', 'Time', 'Model_name']

# Path to store obtained results
os.chdir("../Pareto_results")

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

            data.append((model_name, Mean, sdv, Mean, Time, i))

        for mode in range(int(len(name)/2)):
            # Generate a list of points that will conform the pareto points
            # For each point in data we get its id (iteration), mape and %train
            pareto_points = []
            for i in range(number_models):
                pareto_points.append({"Model name": data[i][0], name[2*mode]: data[i][mode*2+1], name[2*mode+1]: data[i][mode*2+2], "index": data[i][5]})

            # Sort the points based on its mape and % train
            pareto_points.sort(key=lambda point: (point[name[2*mode]], point[name[2*mode+1]]))

            print(f"Pareto {name[2*mode]} vs {name[2*mode+1]}, {len(pareto_points)} total of pareto points")

            # Now we find with points conform the Pareto-optimal points, aka. the pareto frontier
            pareto_frontier = [pareto_points[0]]  # The first point is Pareto optimal by default
            for point in pareto_points[1:]:
                is_pareto = True
                to_remove = []
                # Pareto analysis, one data point is considered better than another
                # if it is at least as good in all dimensions and strictly better
                # in at least one dimension.
                # This is called Pareto dominance.
                for pf_point in pareto_frontier:
                    if point[name[2*mode]] >= pf_point[name[2*mode]] and point[name[2*mode+1]] >= pf_point[name[2*mode+1]]:
                        is_pareto = False
                        break
                    if pf_point[name[2*mode]] >= point[name[2*mode]] and pf_point[name[2*mode+1]] >= point[name[2*mode+1]]:
                        to_remove.append(pf_point)
                if is_pareto:
                    pareto_frontier = [p for p in pareto_frontier if p not in to_remove]
                    pareto_frontier.append(point)

            # Store also the non pareto frontier points
            no_pareto_frontier = [point for point in pareto_points if point not in pareto_frontier]

            # Extract pareto frontier x and y values as well as ids for plotting
            pareto_frontier_x_values = [point[name[2*mode]] for point in pareto_frontier]
            pareto_frontier_y_values = [point[name[2*mode+1]] for point in pareto_frontier]
            pareto_frontier_names = [point["Model name"] for point in pareto_frontier]
            pareto_frontier_ids = [point["index"] for point in pareto_frontier]  # Extract IDs as strings

            # Extract non pareto frontier x and y values for plotting
            no_pareto_frontier_x_values = [point[name[2*mode]] for point in no_pareto_frontier]
            no_pareto_frontier_y_values = [point[name[2*mode+1]] for point in no_pareto_frontier]
            no_pareto_frontier_names = [point["Model name"] for point in no_pareto_frontier]
            no_pareto_frontier_ids = [point["index"] for point in no_pareto_frontier]  # Extract IDs as strings

            # Extract selected frontier x and y
            selected_point = 0
            dummy = 0
            if model == 'Top-power':
                for point in pareto_points:
                    if point['index'] == 6:
                        selected_point = dummy
                    dummy += 1
            elif model == 'Bottom-power':
                for point in pareto_points:
                    if point['index'] == 5:
                        selected_point = dummy
                    dummy += 1
            elif model == 'Time':
                for point in pareto_points:
                    if point['index'] == 0:
                        selected_point = dummy
                    dummy += 1

            selected_frontier_x_value = pareto_points[selected_point][name[2*mode]]
            selected_frontier_y_value = pareto_points[selected_point][name[2*mode+1]]
            selected_frontier_name = pareto_points[selected_point]["Model name"]
            selected_frontier_id = pareto_points[selected_point]["index"]

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
            no_pareto_scatter = axis.scatter(no_pareto_frontier_x_values, no_pareto_frontier_y_values, label='No Pareto Frontier', color='y', edgecolors='black', marker='o', s=20)#s=25)
            pareto_scatter = axis.scatter(pareto_frontier_x_values, pareto_frontier_y_values, label='Pareto Frontier', color='r', edgecolors='black', marker='s')
            selected_scatter = axis.scatter(selected_frontier_x_value, selected_frontier_y_value, label='Selected point', color='b', edgecolors='black', marker='s')


            #########################################
            ## Local functions for cursor acctions ##
            #########################################

            def get_iteration_info(sel_index):
                "Generate a string with the information of this particular iteration"
                # Get the iteration to which the user is pointing with the cursor
                index = pareto_frontier_ids[sel_index]

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
            cursor1 = mplcursors.cursor(pareto_scatter, hover=True)
            cursor1.connect("add", cursor_annotation)

            # Disable hover so when the user hovers the mouse over a point nothing happens
            # But when clicking the info of that iteration is printed
            cursor2 = mplcursors.cursor(pareto_scatter, hover=False)
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

            plt.savefig("Pareto_" + model + "_" + rt + "_sr1_" + name[2*mode] + "_vs_" + name[2*mode+1] + ".tiff")
            plt.show()
        print("\n")
exit()


