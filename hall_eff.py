import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import matplotlib


#___________________________________________________________________________________________________________________________________________

file_name_ip = f"<File path>"    # Si_ntype  # Ge_ptype  # Ge_ptype
precision_format = ".2f"  # Change this to ".2f", ".5f", etc. for different precision
material = "Si-n type"
#___________________________________________________________________________________________________________________________________________

data = pd.read_excel(file_name_ip)


#___________________________________________________________________________________________________________________________________________


            # Least Count
# sigmax = float(input("Least count in x: "))
# sigmay =  float(input("Least count in y: ")) 
sigmax = 0.01
sigmay =  0.1


x = data['Current(A)'] 
y = data['Hall Voltage(mV)']




#___________________________________________________________________________________________________________________________________________




x2=[]   # empty list of square x
xy=[]   # empty list of xy

N=len(x) # number of elements of x or number of observations

for i in range (0,N):
    x2_=x[i]**2       # squaring each elements.... elements of x2 list 
    x2_round = round(x2_, 9)   # Rounding up
    x2.append (x2_round)   # adding to the empty list

    xy_=x[i]*y[i]     # multiplying x and y be index
    xy_round = round(xy_, 9)   # Rounding up 
    xy.append (xy_round)   # adding to the empty list.... elements of xy list 


print()
# print (f'The LIST OF X IS:\n {x}')
# print (f'The LIST OF y IS:\n {y}')
# print()
# print (f'THE LIST OF X SQUARE IS:\n    {x2}')
# print (f'THE LIST OF XY IS: \n    {xy}')

sum_x=sum(x)
sum_y=sum(y)

sum_x2= sum(x2)
sum_xy= sum(xy)

print (f'N = {N}')
# print (f'THE SUMATION OF X LIST IS:   {sum_x}')
# print (f'THE SUMATION OF Y LIST IS: {sum_y}')
# print (f'THE SUMATION OF X SQUARE LIST IS:    {sum_x2}')
# print (f'THE SUMATION OF XY LIST IS:    {sum_xy}')

#___________________________________________________________________________________________________________________________________________
x_values = x
y_values = y
x_squared = x2
xy_values = xy



# OUtput Table Format

# Convert to regular floats for easier handling
x_values = [float(x) for x in x_values]
y_values = [float(y) for y in y_values]

# Compute x^2 and xy
x_squared = [x ** 2 for x in x_values]
xy_values = [x * y for x, y in zip(x_values, y_values)]

# for i in range(0)

# vacant = ["..."+""*n for n in range(0,len(x_values)) ]


# Create a DataFrame to format as a table
df = pd.DataFrame({
    'x': x_values,
    'y': y_values,
    'x^2': x_squared,
    'xy': xy_values
})

# Calculate the sum for each column
sum_row = pd.DataFrame(df.sum()).T


# Define column widths
col_widths = [10, 10, 15, 15]

# Function to format the header and rows
def format_row(x, y, x2, xy, col_widths, precision):
    """Formats a row with proper spacing and alignment based on column widths."""
    return f"{x:<{col_widths[0]}{precision}} {y:<{col_widths[1]}{precision}} {x2:<{col_widths[2]}{precision}} {xy:<{col_widths[3]}{precision}}"

# Create the header
header = f"{'x':<{col_widths[0]}} {'y':<{col_widths[1]}} {'x^2':<{col_widths[2]}} {'xy':<{col_widths[3]}}"

# Format each row of the DataFrame
formatted_rows = [format_row(row['x'], row['y'], row['x^2'], row['xy'], col_widths, precision_format) for _, row in df.iterrows()]

# Format the sum row
formatted_sum = format_row(sum_row.iloc[0]['x'], sum_row.iloc[0]['y'], sum_row.iloc[0]['x^2'], sum_row.iloc[0]['xy'], col_widths, precision_format)

# Create the separators
underscore_separator = "_" * (sum(col_widths) + len(col_widths) - 1)  # line of underscores for normal rows
equals_separator = "=" * (sum(col_widths) + len(col_widths) - 1)  # line of equals for sum row

# Combine everything into the final table output
final_output = f"{header}\n{underscore_separator}\n" + f"\n{underscore_separator}\n".join(formatted_rows) + f"\n{equals_separator}\n" + formatted_sum

print()
# Display the final output
print(final_output)



#___________________________________________________________________________________________________________________________________________




# FORMULAE OF INTERCEPT AND SLOPE.......

#  delta = N * summation of x square - square of (summation of x)
#  slope = ( N * summation of xy - summation x * summation of y )/delta
#  intercept = ( summation x2 * summation y - summation x * summation of xy )/delta

delta = N*sum_x2 - sum_x**2

slope = (N*sum_xy - sum_x*sum_y)/delta
intercept = (sum_x2*sum_y - sum_x*sum_xy)/delta

print()
print (f'DELTA IS: {delta}')
print (f'SLOPE OF GRAPH IS:  {slope}')
print (f'INTERCEPT OF GRAPH IS:   {intercept}')



#___________________________________________________________________________________________________________________________________________

# FORMULAE OF ERRORS.......

#  delta_y = root over [1/(N-2) * {summation of ( y[i] - intercept - slope * x[i] )**2 }]
#  delta _intercept = delta_y * root over [summation of x square / delta]
#  delta_slope = delta_y * root over [N / delta]



inner_element_sq_list=[]    # empty list
inner_element_list=[]     # empty list 
for i in range (0,N):
    inner_element = y[i] - intercept - x[i]*slope
    inner_element_round = round (inner_element,9)
    inner_element_sq = inner_element_round**2
    inner_element_sq_list.append(inner_element_sq)
    inner_element_list.append(inner_element_round)

print()
print()


sum_inner_element_sq = sum (inner_element_sq_list)
inner_root = sum_inner_element_sq/(N-2)


#___________________________________________________________________________________________________________________________________________


# OUtput Table Format
y_a_bx_values = inner_element_list
squared_diff = inner_element_sq_list

# Create a DataFrame to format as a table
df = pd.DataFrame({
    'x': x_values,
    'y': y_values,
    'y - a - bx': y_a_bx_values,
    '(y - a - bx)^2': squared_diff
})

# Calculate the sum for each column
sum_row = pd.DataFrame(df.sum()).T

col_widths = [10, 10, 15, 20]

# precision_format = ".9f"  # Change this to ".2f", ".5f", etc. for different precision

# Function to format the header and rows with dynamic precision
def format_row(x, y, y_abx, sq_diff, col_widths, precision):
    """Formats a row with proper spacing and alignment based on column widths and precision."""
    return f"{x:<{col_widths[0]}{precision}} {y:<{col_widths[1]}{precision}} {y_abx:<{col_widths[2]}{precision}} {sq_diff:<{col_widths[3]}{precision}}"

# Create the header
header = f"{'x':<{col_widths[0]}} {'y':<{col_widths[1]}} {'y - a - bx':<{col_widths[2]}} {'(y - a - bx)^2':<{col_widths[3]}}"

# Format each row of the DataFrame
formatted_rows = [format_row(row['x'], row['y'], row['y - a - bx'], row['(y - a - bx)^2'], col_widths, precision_format) for _, row in df.iterrows()]

# Format the sum row
formatted_sum = format_row(sum_row.iloc[0]['x'], sum_row.iloc[0]['y'], sum_row.iloc[0]['y - a - bx'], sum_row.iloc[0]['(y - a - bx)^2'], col_widths, precision_format)

# Create the separators
underscore_separator = "_" * (sum(col_widths) + len(col_widths) - 1)  # line of underscores for normal rows
equals_separator = "=" * (sum(col_widths) + len(col_widths) - 1)  # line of equals for sum row

# Combine everything into the final table output
final_output = f"{header}\n{underscore_separator}\n" + f"\n{underscore_separator}\n".join(formatted_rows) + f"\n{equals_separator}\n" + formatted_sum

print()
# Display the final output
print(final_output)


#___________________________________________________________________________________________________________________________________________


# Modify delta_y as you need...

delta_y = inner_root**0.5
# delta_y = sigmay

gama_intercept = (sum_x2/delta)**0.5
delta_intercept = delta_y*gama_intercept

gama_slope = (N/delta)**0.5
delta_slope = delta_y*gama_slope

print()
print()
print (f'DELTA Y OF GRAPH :    {delta_y}')
print (f'DELTA INTERCEPT OF GRAPH :    {delta_intercept}')
print (f'DELTA SLOPE OF GRAPH :   {delta_slope}')



#___________________________________________________________________________________________________________________________________________


print("\n\n\n\t\t\t\tok")

#___________________________________________________________________________________________________________________________________________







def create_graph(file_name):

    global x, y

    # Perform linear regression to get slope and intercept
    slope, intercept = np.polyfit(x, y, 1)

    global sigmay, sigmax, delta_slope

    # Create the plot
    plt.figure(figsize=(9, 6))  # 9 inches wide and 6 inches tall
    plt.errorbar(x, y, yerr=sigmay, xerr=sigmax, capsize=3, fmt='o', markersize= 2, color= 'blue', ecolor='#949494', label=f'Data points with errorbar')
    

    if intercept < 0:
        plt.plot(x, slope*x + intercept, color='r', label=f'Fitted Line: y = {slope:.4e}x - {-1*intercept:.1e}')

    else:
        plt.plot(x, slope*x + intercept, color='r', label=f'Fitted Line: y = {slope:.4e}x + {intercept:.1e}')    #{intercept:.1e}


    plt.plot([], [], color="none",label=f"Error in slope: {delta_slope:.3e}")

    # Set labels, title, and legend

    xlabel_ = r"Current(A)"
    ylabel_ = r"Hall Voltage(mV)"
    title_ = f"Hall coeffiecient for {material} curve"
    
    # xlabel = r"f(Hz)"
    # ylabel = "slope"
    # plt.xlabel(r'$(\dfrac{1}{n^2} − \dfrac{1}{m^2})$')
    # plt.xlabel(r'$\lambda_{\text{observed}}$')
    # plt.ylabel(r'$\lambda_{\text{given}}$(nm)')
    # plt.title(r"$\dfrac{1}{\lambda}$ vs $(\dfrac{1}{n^2} − \dfrac{1}{n^2}) plot for the Balmer lines")

    plt.xlabel(xlabel_ ,fontsize=12)
    plt.ylabel(ylabel_ ,fontsize=12)
    # plt.title(title_)

    

    
    plt.legend(loc='best',fontsize=12)

    # Add grid lines
    plt.grid(True)

    # # Set axis limits
    # plt.xlim(min(x)-1, max(x) + 1)
    # plt.ylim(min(y)-1, max(y) + 1)

    # # Add arrowheads to axes
    # plt.annotate("", xy=(max(x) + 0.5, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))
    # plt.annotate("", xy=(0, max(y) + 0.5), xytext=(0, 0), arrowprops=dict(arrowstyle="->"))

    # # Add axis labels
    # plt.text(max(x) + 1, -0.2, 'X', fontsize=16)
    # plt.text(-0.5, max(y) + 4, 'Y', fontsize=16)

    # plt.savefig(f"<FILE PATH>", bbox_inches='tight', pad_inches=0.1)  
    # plt.close()

    plt.show()

create_graph(file_name_ip)





# print(  "delta_y = root over [1/(N-2) * {summation of (y-a-bx)^2 }]")
# print( "delta _intercept = delta_y * root over [summation of x^2 / delta]")
# print( "delta_slope = delta_y * root over [N / delta]")

