import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk
from tkinter import messagebox

#External state recidivism rates Source: https://csgjusticecenter.org/wp-content/uploads/2024/04/50-States-1-Goal_For-PDF_with508report.pdf
state_recidivism_rates = {
    'Arizona': 0.38,
    'California': 0.64,
    'Colorado': 0.52,
    'Connecticut': 0.54,
    'Florida': 0.28,
    'Georgia': 0.27,
    'Hawaii': 0.48,
    'Iowa': 0.32,
    'Kansas': 0.34,
    'Kentucky': 0.30,
    'Louisiana': 0.37,
    'Maine': 0.25,
    'Massachusetts': 0.39,
    'Michigan': 0.32,
    'Minnesota': 0.26,
    'Missouri': 0.44,
    'Nebraska': 0.26,
    'Nevada': 0.27,
    'New Jersey': 0.35,
    'New York': 0.40,
    'North Carolina': 0.25,
    'North Dakota': 0.40,
    'Ohio': 0.31,
    'Oklahoma': 0.23,
    'Oregon': 0.16,
    'Pennsylvania': 0.43,
    'South Carolina': 0.31,
    'South Dakota': 0.45,
    'Tennessee': 0.49,
    'Texas': 0.22,
    'Washington': 0.28,
    'West Virginia': 0.29,
    'Wisconsin': 0.39,
    'Wyoming': 0.23
}


# Recidivism rates by year 3 from Table 4 "Cumulative percent of state prisoners released in 34 states in 2012 who were arrested following release, by sex, race or ethnicity, age at release, and year following release"
recidivism_rates_by_characteristic_year_3_after_release = {
    'Male': 0.617,
    'Female': 0.529,
    'White': 0.598,
    'Black': 0.644,
    'Hispanic': 0.594,
    'American Indian/Alaska Native': 0.689,
    'Asian': 0.573,
    'Other': 0.593,
    '24 or younger': 0.723,
    '25-39': 0.648,
    '40 or older': 0.518,
}


#Proportion of number of state prisoners realeased in 34 States who were included in study sample, Appendix Table 1
state_release_proportion = {
    'Arizona': 0.035,
    'California': 0.115,
    'Colorado': 0.0247,
    'Connecticut': 0.0237,
    'Florida': 0.0798,
    'Georgia': 0.0406,
    'Hawaii': 0.005,
    'Iowa': 0.0122,
    'Kansas': 0.008,
    'Kentucky': 0.034,
    'Louisiana': 0.033,
    'Maine': 0.002,
    'Massachusetts': 0.006,
    'Michigan': 0.030,
    'Minnesota': 0.016,
    'Missouri': 0.039,
    'Nebraska': 0.006,
    'Nevada': 0.013,
    'New Jersey': 0.026,
    'New York': 0.055,
    'North Carolina': 0.033,
    'North Dakota': 0.002,
    'Ohio': 0.039,
    'Oklahoma': 0.017,
    'Oregon': 0.012,
    'Pennsylvania': 0.048,
    'South Carolina': 0.019,
    'South Dakota': 0.005,
    'Tennessee': 0.041,
    'Texas': 0.131,
    'Washington': 0.018,
    'West Virginia': 0.007,
    'Wisconsin': 0.020,
    'Wyoming': 0.002
}


#Use Beta Distrbution to model proportions
#This will draw a possible probability from the Beta distribution, reflecting uncertainty instead of just a point estimate, sample size from data study of 92100
#Since our sample size is very large, we are pretty confident about our given probability, but this can help future smaller samples!
def likelihood(characterisitc, sample_size=92100, confidence=0.95):
    if characterisitc not in recidivism_rates_by_characteristic_year_3_after_release:
        raise ValueError(f"Characteristic '{characterisitc}' not found in recidivism rates.")
     
    rate = recidivism_rates_by_characteristic_year_3_after_release[characterisitc]
    alpha = rate * sample_size
    beta_param = (1 - rate) * sample_size
    mode = (alpha - 1) / (alpha + beta_param - 2) if alpha > 1 and beta_param > 1 else rate
    # If mode cannot be calculated, fall back to the average
    if alpha <= 1 or beta_param <= 1:
        average = rate
    else:
        average = mode
    return average

#Compute P(Recidivism | State, age, race, sex) = P(State, age, race, sex | Recidivism) * P(Recidviism | State) / P(State, age, race, sex)
def posterior_recidivism_by_state_given_optional_age_race_sex(state, age_group=None, race=None, sex=None):
    if state not in state_recidivism_rates:
        raise ValueError(f"State '{state}' not found in recidivism rates.")
    if not any([age_group, race, sex]):
        raise ValueError("At least one of 'age_group', 'race', or 'sex' must be provided.")
    
    prior = state_recidivism_rates[state]  #P(Recidivism | State)
    
    likelihoods = []
    if age_group:
        likelihoods.append(likelihood(age_group))
    if race:
        likelihoods.append(likelihood(race))
    if sex:
        likelihoods.append(likelihood(sex))
    
    #multiply all likelihoods together (assuming independence)
    likelihood_value = 1 if not likelihoods else np.prod(likelihoods)
    
    #compute evidence (normalization)
    evidence = likelihood_value * prior + (1 - prior) * (1 - likelihood_value)
    
    #Bayes' theorem
    posterior = (likelihood_value * prior) / evidence
    
    return posterior


#Compute P(State | Recidivism, Race, Age, Sex) = P(Recidivism, Race, Age, Sex | State) * P(State) / P(Recidivsm, Race, Age, Sex)
def posterior_state_given_recidivism(race=None, age=None, sex=None):
    posterior_probs = {}

    #P(Recidivism | Race or Age or Sex) from national data
    P_recidivism_given_race = recidivism_rates_by_characteristic_year_3_after_release.get(race, 1)
    P_recidivism_given_age = recidivism_rates_by_characteristic_year_3_after_release.get(age, 1)
    P_recidivism_given_sex = recidivism_rates_by_characteristic_year_3_after_release.get(sex, 1)

    #Combined recidivism rate considering race, age, and sex
    P_recidivism_given_race_age_sex = (P_recidivism_given_race + P_recidivism_given_age + P_recidivism_given_sex) / 3

    #Compute denominator: Sum over all states
    normalization_factor = sum(
        state_recidivism_rates[state] * state_release_proportion[state]
        for state in state_recidivism_rates
    )

    for state in state_recidivism_rates:
        # Estimate P(Recidivism | State, Race)
        likelihood = P_recidivism_given_race_age_sex * state_recidivism_rates[state]

        #Prior: P(State)
        prior = state_release_proportion[state]

        #Bayes' theorem: P(State | Recidivism, Race)
        posterior = (likelihood * prior) / normalization_factor

        posterior_probs[state] = posterior

    return posterior_probs

#Data with Prior Arrests, Find the Critical Point of number of Arrests in Recidivists
#Data points from Table 6 "Cumulative percent of state prisoners released in 34 states in 2012 who were arrested following release, by number of prior arrests, age at first arrest, and year following release"
def prior_arrests_and_recidivism():
    prior_arrests = np.array([2, 3, 4, 5, 9, 10])
    recidivism_rates = np.array([37.9, 45.0, 55.0, 59.3, 62, 73.2])
    #plot
    plt.scatter(prior_arrests, recidivism_rates, label="Data")
    plt.xlabel("Number of Prior Arrests")
    plt.ylabel("Three Recidivism Rate (%)")
    plt.title("Impact of Prior Arrests on Recidivism")
    
    #best fit into a regression line
    z = np.polyfit(prior_arrests, recidivism_rates, 2)  # Quadratic fit
    p = np.poly1d(z)
    plt.plot(prior_arrests, p(prior_arrests), "r--", label="Trend Line")
    plt.legend()
    plt.show()
    
    #Compare Impact of Age vs. Prior Arrests
    #Which has a stronger impact: age at first arrest or prior arrests?
    #Perform a correlation analysis to compare.
    prior_arrests = np.array([2, 3.5, 7, 10])  # Midpoints for ranges
    recidivism_rates_prior = np.array([47.6, 60.3, 70.0, 81.0])  #Recidivism at Year 5
    
    age_at_first_arrest = np.array([17, 18.5, 22, 27, 32, 37, 42])  #Midpoints of age
    recidivism_rates_age = np.array([80.0, 75.6, 65.9, 57.7, 49.0, 42.5, 29.1])  #Recidivism at Year 5
    
    #Get correlations with scipy pearsonr
    corr_prior, _ = stats.pearsonr(prior_arrests, recidivism_rates_prior)
    corr_age, _ = stats.pearsonr(age_at_first_arrest, recidivism_rates_age)
    
    print(f"Correlation (Prior Arrests vs. Recidivism): {corr_prior:.2f}")
    print(f"Correlation (Age at First Arrest vs. Recidivism): {corr_age:.2f}")
    
    #Plot the data
    plt.figure(figsize=(10,5))
    plt.scatter(prior_arrests, recidivism_rates_prior, label="Prior Arrests vs. Recidivism", color="red")
    plt.scatter(age_at_first_arrest, recidivism_rates_age, label="Age at First Arrest vs. Recidivism", color="blue")
    plt.xlabel("Variable (Prior Arrests / Age at First Arrest)")
    plt.ylabel("Recidivism Rate (%)")
    plt.title("Comparing Impact of Prior Arrests vs. Age at First Arrest on Recidivism")
    plt.legend()
    plt.show()


#Bootstrap function for correlation
def bootstrap_correlation(x, y, n_bootstrap=1000):
    correlations = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x), size=len(x), replace=True)
        sample_x, sample_y = x[idx], y[idx]

        if np.std(sample_x) == 0 or np.std(sample_y) == 0:  
            continue  #Skip cases where there's no variation

        correlations.append(stats.pearsonr(sample_x, sample_y)[0])
    
    return np.percentile(correlations, [2.5, 97.5])

#Get 95% confidence interval for correlation
#corr_ci = bootstrap_correlation(prior_arrests, recidivism_rates_prior)
#print(f"95% Confidence Interval for Correlation: ({corr_ci[0]:.2f}, {corr_ci[1]:.2f})")


# ----------------------------------------------- GUI ---------------------------------------- #
# Function to handle the 'Next' button click and move to the main GUI
def on_next_button_click():
    # Destroy the welcome screen
    welcome_screen.destroy()

    # Create the main window
    create_main_window()

# Function to handle calculation on the 'Calculate' button click
def p_recidivism_given_characteristics_button_click():
    # Get the user input
    state = state_input.get()
    age_group = age_input.get()
    race = race_input.get()
    sex = sex_input.get()

    # Calculate the posterior probability based on the recidivism data
    try:
        posterior = posterior_recidivism_by_state_given_optional_age_race_sex(state, age_group, race, sex)
        result_label.config(text=f"P(Recidivism | State, Age Group, Race, Sex): {posterior:.4f}", font=("Arial", 14, "italic"))
    except ValueError as e:
        messagebox.showerror("Error", str(e))

# Function to handle state probability calculation based on recidivism characteristics
def p_state_recidivism_and_characteristics_button_click():
    # Get the user input
    race = race_input.get()
    age_group = age_input.get()
    sex = sex_input.get()

    # Calculate the posterior probabilities for each state
    posterior_probs = posterior_state_given_recidivism(race, age_group, sex)
    
    # Display the results
    sorted_probs = sorted(posterior_probs.items(), key=lambda x: x[1], reverse=True)
    result_text = "P(State | Recidivism, Race, Age, Sex):\n"
    for state, prob in sorted_probs:
        result_text += f"{state}: {prob:.4f}\n"
    result_label.config(text=result_text, font=("Arial", 10, "italic"))



# Function to create the main window after clicking "Next"
def create_main_window():
    global state_input, age_input, race_input, sex_input, result_label

    # Create the main window (using the same Tk instance)
    main_window = tk.Tk()
    main_window.title("Probabilistic Recidivism Model")
    main_window.geometry("600x400")  # Set the window size to 600x400
    main_window.config(bg="lightblue")

    # Create and place the widgets with better padding and alignment
    tk.Label(main_window, text="State:", font=("Arial", 12, "bold"), bg="lightblue").grid(row=0, column=0, padx=10, pady=10, sticky="e")
    state_input = tk.Entry(main_window, font=("Arial", 12), width=30)
    state_input.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(main_window, text="Age Group:", font=("Arial", 12, "bold"), bg="lightblue").grid(row=1, column=0, padx=10, pady=10, sticky="e")
    age_input = tk.Entry(main_window, font=("Arial", 12), width=30)
    age_input.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(main_window, text="Race:", font=("Arial", 12, "bold"), bg="lightblue").grid(row=2, column=0, padx=10, pady=10, sticky="e")
    race_input = tk.Entry(main_window, font=("Arial", 12), width=30)
    race_input.grid(row=2, column=1, padx=10, pady=10)

    tk.Label(main_window, text="Sex:", font=("Arial", 12, "bold"), bg="lightblue").grid(row=3, column=0, padx=10, pady=10, sticky="e")
    sex_input = tk.Entry(main_window, font=("Arial", 12), width=30)
    sex_input.grid(row=3, column=1, padx=10, pady=10)

    # Buttons with better padding and larger size
    tk.Button(main_window, text="P(Recidivism | State, Age Group, Race, Sex)", command=p_recidivism_given_characteristics_button_click, font=("Arial", 12), bg="lightgreen").grid(row=0, column=2, padx=10, pady=15)
    tk.Button(main_window, text="P(State | Recidivism, Race, Age, Sex)", command=p_state_recidivism_and_characteristics_button_click, font=("Arial", 12), bg="lightcoral").grid(row=1, column=2, padx=10, pady=15)

    # Button to move to the plotting screen
    tk.Button(main_window, text="Prior Arrests and Recidivism Graphs", command=prior_arrests_and_recidivism, font=("Arial", 12), bg="lightblue").grid(row=2, column=2, columnspan=2, padx=10, pady=15)

    # Result Label with a more prominent font
    # Result Label with a more prominent font
    result_label = tk.Label(main_window, text="Results will be shown here.", font=("Arial", 14, "italic"), bg="lightblue", justify="left", anchor="w")
    result_label.grid(row=0, column=4, rowspan=6, padx=10, pady=15, sticky="w")
    
    # Run the main window
    main_window.mainloop()

# Create the welcome screen (this is the root window)
welcome_screen = tk.Tk()
welcome_screen.title("Welcome")
welcome_screen.geometry("600x400")  # Set the window size to 600x400
welcome_screen.config(bg="lightblue")

# Display a welcome message
welcome_label = tk.Label(welcome_screen, text="Welcome to the Probabilistic Recidivism Model\n\nClick Next to proceed.")
welcome_label.pack(pady=20)

# Next button to move to the main GUI
next_button = tk.Button(welcome_screen, text="Next", command=on_next_button_click)
next_button.pack(pady=10)

# Start the welcome screen (root window)
welcome_screen.mainloop()
# ----------------------------------------------- GUI ---------------------------------------- #