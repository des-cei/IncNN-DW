#!/usr/bin/env python3

"""
Visualize grid search information from a JSON file

Author      : Juan Encinas <juan.encinas@upm.es>
Date        : October 2023
Description : This script reads a JSON file containing information about a grid
              search process and plots the error metrics and percentage of
              trained obsevations of the top, bottom and time models when
              varying certain parameters defined by the user in a configuration
              file "config/config.json".
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import mplcursors
import os
import pickle

# Path to store obtained results
os.chdir("Results_NN/Complete_dictionaries")

with open('NN-3layers-dictionary_pareto.pkl', 'rb') as handle:
    results = pickle.load(handle)
handle.close()


model_type = ['Top-power', 'Bottom-power', 'Time']
results_types = ['MAPE-mean-error', 'MAPE-sdv-error', 'MAPE-mean-error_cv', 'Train-time', 'Infer-time', 'Model_name']

# Path to store obtained results
os.chdir("../Paretos/3LNN")

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
name = ["MAPE mean (%)", "MAPE sdv (%)", "Total train time (s)", "Individual test time (s)", "MAPE mean (%)", "Train + test time (s)", "MAPE mean cv (%)", "Train + test time (%)"]
number_models = len(results["Top-power"])

# Iterate over each model type
for model in model_type:
    if model == "Top-power":
        os.chdir("Top_Power")
    elif model == "Bottom-power":
        os.chdir("../Bottom_Power")
    elif model == "Time-power":
        os.chdir("../Time")
    # Print model type in terminal
    print("\n{}".format(model.replace('_', ' ').capitalize()))
    data = []

    for i in range(number_models):
        MAPE_mean = float(results[model][str(i)][results_types[0]]) # MAPE mean
        MAPE_sdv = float(results[model][str(i)][results_types[1]]) # MAPE sdv
        MAPE_mean_cv = float(results[model][str(i)][results_types[2]]) # MAPE mean cv
        train_time = float(results[model][str(i)][results_types[3]]) # Train time
        infer_time = float(results[model][str(i)][results_types[4]]) # Test time
        model_name = results[model][str(i)][results_types[5]] # Model name
        total_time = train_time + infer_time * 98525

        data.append((model_name, MAPE_mean, MAPE_sdv, train_time, infer_time, MAPE_mean, total_time, MAPE_mean_cv, total_time, i))

    for mode in range(len(name)/2):
        # Generate a list of points that will conform the pareto points
        # For each point in data we get its id (iteration), mape and %train
        pareto_points = []
        for i in range(number_models):
            pareto_points.append({"Model name": data[i][0], name[2*mode]: data[i][mode*2+1], name[2*mode+1]: data[i][mode*2+2], "index": data[i][9]})

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

        # Extract pareto frontier x and y values for plotting
        no_pareto_frontier_x_values = [point[name[2*mode]] for point in no_pareto_frontier]
        no_pareto_frontier_y_values = [point[name[2*mode+1]] for point in no_pareto_frontier]
        no_pareto_frontier_names = [point["Model name"] for point in no_pareto_frontier]
        no_pareto_frontier_ids = [point["index"] for point in no_pareto_frontier]  # Extract IDs as strings

        # Extract selected frontier x and y
        #if model == 'Time':
        selected_point = 0
        dummy = 0
        if model == 'Top-power':
            for point in pareto_points:
                if point['index'] == 16:
                    selected_point = dummy
                dummy += 1
        elif model == 'Bottom-power':
            for point in pareto_points:
                if point['index'] == 34:
                    selected_point = dummy
                dummy += 1
        elif model == 'Time':
            for point in pareto_points:
                if point['index'] == 17:
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
            plot_title = "Top Power Model"
        elif model == "Bottom-power":
            plot_title = "Bottom Power Model"
        else:
            plot_title = "Time Model"

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
                if (data[i][9] == index):
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
            resulting_string += "MAPE mean cv: {:.3f} %\n".format(data[actual_index][3])
            resulting_string += "Train time: {:.3f} s\n".format(data[actual_index][4])
            resulting_string += "Test time: {:.3f} ms\n".format(data[actual_index][5]*1000)
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

        plt.savefig("Pareto_" + model + "_" + name[2*mode] + "_vs_" + name[2*mode+1] + ".tiff")
        plt.show()
    print("\n")
exit()


