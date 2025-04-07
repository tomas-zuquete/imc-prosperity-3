import itertools

# Define the assets
assets = ['Snowballs', 'Pizza\'s', 'Silicon Nuggets', 'SeaShells']
start_asset = 'SeaShells'
intermediate_assets = ['Snowballs', 'Pizza\'s', 'Silicon Nuggets']

# Define the exchange rate table as a nested dictionary
# rates[from_currency][to_currency] = exchange_rate
rates = {
    'Snowballs': {
        'Snowballs': 1,
        'Pizza\'s': 1.45,
        'Silicon Nuggets': 0.52,
        'SeaShells': 0.72
    },
    'Pizza\'s': {
        'Snowballs': 0.70, # Corrected based on the previous analysis S->P->B factor
        'Pizza\'s': 1,
        'Silicon Nuggets': 0.31,
        'SeaShells': 0.48
    },
    'Silicon Nuggets': {
        'Snowballs': 1.95,
        'Pizza\'s': 3.10,
        'Silicon Nuggets': 1,
        'SeaShells': 1.49
    },
    'SeaShells': {
        'Snowballs': 1.34,
        'Pizza\'s': 1.98,
        'Silicon Nuggets': 0.64,
        'SeaShells': 1
    }
}

# --- Verification of Rates (Optional but recommended) ---
# Check if converting back roughly equals 1/rate (allowing for some rounding in the table)
# print(f"S->P: {rates['SeaShells']['Pizza\'s']}, P->S: {rates['Pizza\'s']['SeaShells']}, Product: {rates['SeaShells']['Pizza\'s'] * rates['Pizza\'s']['SeaShells']:.4f}")
# print(f"S->B: {rates['SeaShells']['Snowballs']}, B->S: {rates['Snowballs']['SeaShells']}, Product: {rates['SeaShells']['Snowballs'] * rates['Snowballs']['SeaShells']:.4f}")
# print(f"S->N: {rates['SeaShells']['Silicon Nuggets']}, N->S: {rates['Silicon Nuggets']['SeaShells']}, Product: {rates['SeaShells']['Silicon Nuggets'] * rates['Silicon Nuggets']['SeaShells']:.4f}")
# print(f"P->B: {rates['Pizza\'s']['Snowballs']}, B->P: {rates['Snowballs']['Pizza\'s']}, Product: {rates['Pizza\'s']['Snowballs'] * rates['Snowballs']['Pizza\'s']:.4f}")
# print(f"P->N: {rates['Pizza\'s']['Silicon Nuggets']}, N->P: {rates['Silicon Nuggets']['Pizza\'s']}, Product: {rates['Pizza\'s']['Silicon Nuggets'] * rates['Silicon Nuggets']['Pizza\'s']:.4f}")
# print(f"B->N: {rates['Snowballs']['Silicon Nuggets']}, N->B: {rates['Silicon Nuggets']['Snowballs']}, Product: {rates['Snowballs']['Silicon Nuggets'] * rates['Silicon Nuggets']['Snowballs']:.4f}")
# --- End Verification ---


max_profit_factor = 1.0  # Start with 1.0, meaning no profit
optimal_sequence = None
start_capital = 2_000_000

print("Searching for optimal trading loops...")

# Loop through the number of intermediate steps (1 to 4, but max is len(intermediate_assets))
# k = number of intermediate assets in the loop
max_intermediate_steps = min(4, len(intermediate_assets)) # User asked for up to 4 intermediates
for k in range(1, max_intermediate_steps + 1):
    # Generate all permutations of intermediate assets of length k
    for intermediate_perm in itertools.permutations(intermediate_assets, k):
        # Construct the full trading sequence: Start -> Intermediates -> Start
        current_sequence = [start_asset] + list(intermediate_perm) + [start_asset]
        
        # Calculate the profit factor for this sequence
        current_factor = 1.0
        valid_sequence = True
        for i in range(len(current_sequence) - 1):
            from_curr = current_sequence[i]
            to_curr = current_sequence[i+1]
            
            # Ensure the rate exists (should always be true with this setup)
            if from_curr in rates and to_curr in rates[from_curr]:
                current_factor *= rates[from_curr][to_curr]
            else:
                print(f"Warning: Rate not found for {from_curr} -> {to_curr}")
                valid_sequence = False
                break
        
        if not valid_sequence:
            continue

        # Uncomment to see all profitable loops checked
        # if current_factor > 1.0:
        #    print(f"  Testing sequence: {' -> '.join(current_sequence)}")
        #    print(f"    Profit Factor: {current_factor:.6f}")

        # Check if this sequence is more profitable than the current best
        if current_factor > max_profit_factor:
            max_profit_factor = current_factor
            optimal_sequence = current_sequence
            # print(f"    --- New optimal sequence found ---") # Optional: Track updates

print("\n--- Search Complete ---")

if optimal_sequence:
    final_amount = start_capital * max_profit_factor
    profit = final_amount - start_capital
    
    print(f"Optimal trading sequence found:")
    print(f"  {' -> '.join(optimal_sequence)}")
    print(f"Profit Factor: {max_profit_factor:.8f}")
    print(f"Starting Capital (SeaShells): {start_capital:,.2f}")
    print(f"Ending Capital (SeaShells):   {final_amount:,.2f}")
    print(f"Profit (SeaShells):           {profit:,.2f}")
else:
    print("No profitable trading sequence found (max factor <= 1.0).")

# Example calculation trace for the optimal sequence found previously:
# S -> P -> B -> N -> S
# Factor = rates['SeaShells']['Pizza\'s'] * rates['Pizza\'s']['Snowballs'] * rates['Snowballs']['Silicon Nuggets'] * rates['Silicon Nuggets']['SeaShells']
# Factor = 1.98 * 0.70 * 0.52 * 1.49 
# Factor = 1.0741416
# print(f"\nManual check for S->P->B->N->S: {1.98 * 0.70 * 0.52 * 1.49:.8f}")