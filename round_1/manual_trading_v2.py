import numpy as np
from itertools import permutations

# Define the currencies/assets
assets = ["Snowballs", "Pizza's", "Silicon Nuggets", "SeaShells"]

# Define the exchange rate table as a matrix
# Reading from row to column - e.g., rates[0][1] means trading Snowballs for Pizza's
rates = np.array([
    [1.00, 1.45, 0.52, 0.72],    # From Snowballs to...
    [0.70, 1.00, 0.31, 0.48],    # From Pizza's to...
    [1.95, 3.10, 1.00, 1.49],    # From Silicon Nuggets to...
    [1.34, 1.98, 0.64, 1.00]     # From SeaShells to...
])

def calculate_profit(path, initial_amount=2000000):
    """
    Calculate the profit for a given trading path
    
    Args:
        path: List of indices representing assets to trade through
        initial_amount: Starting amount of the initial asset
    
    Returns:
        final_amount, profit_percentage, path_names
    """
    current_amount = initial_amount
    path_names = [assets[path[0]]]
    
    for i in range(len(path)-1):
        from_idx = path[i]
        to_idx = path[i+1]
        current_amount *= rates[from_idx][to_idx]
        path_names.append(assets[to_idx])
    
    profit_percentage = (current_amount / initial_amount - 1) * 100
    return current_amount, profit_percentage, path_names

def find_optimal_path(max_path_length=4, initial_asset_idx=3, initial_amount=2000000):
    """
    Find the optimal trading path that maximizes profit
    
    Args:
        max_path_length: Maximum number of assets to trade through
        initial_asset_idx: Index of the starting asset (3 = SeaShells)
        initial_amount: Starting amount of the initial asset
    
    Returns:
        best_path, best_profit, best_final_amount
    """
    best_profit = 0
    best_path = None
    best_final_amount = initial_amount
    best_path_detail = None
    
    # Only test cycles that start and end with SeaShells
    for path_length in range(2, max_path_length + 2):
        # For paths of length 2, we need special handling to ensure both positions are filled
        if path_length == 2:
            for middle_asset in range(4):
                if middle_asset != initial_asset_idx:  # Skip if it's the same as starting asset
                    path = (initial_asset_idx, middle_asset, initial_asset_idx)
                    
                    # Calculate profit for this path
                    current_amount = initial_amount
                    for i in range(len(path)-1):
                        from_idx = path[i]
                        to_idx = path[i+1]
                        current_amount *= rates[from_idx][to_idx]
                    
                    profit_percentage = (current_amount / initial_amount - 1) * 100
                    
                    if profit_percentage > best_profit:
                        best_profit = profit_percentage
                        best_path = path
                        best_final_amount = current_amount
                        _, _, best_path_detail = calculate_profit(path, initial_amount)
        else:
            # Get all possible paths that start and end with SeaShells
            for middle_assets in permutations([i for i in range(4) if i != initial_asset_idx], path_length - 2):
                # Construct full path: start with SeaShells, go through middle assets, end with SeaShells
                path = (initial_asset_idx,) + middle_assets + (initial_asset_idx,)
                
                # Calculate profit for this path
                current_amount = initial_amount
                for i in range(len(path)-1):
                    from_idx = path[i]
                    to_idx = path[i+1]
                    current_amount *= rates[from_idx][to_idx]
                
                profit_percentage = (current_amount / initial_amount - 1) * 100
                
                if profit_percentage > best_profit:
                    best_profit = profit_percentage
                    best_path = path
                    best_final_amount = current_amount
                    _, _, best_path_detail = calculate_profit(path, initial_amount)
    
    return best_path, best_profit, best_final_amount, best_path_detail

def format_amount(amount):
    """Format a number with commas for thousands"""
    return f"{amount:,.2f}"

# Find the optimal trading path starting with SeaShells (index 3)
best_path, best_profit, best_final_amount, best_path_detail = find_optimal_path(
    max_path_length=4, 
    initial_asset_idx=3,  # SeaShells
    initial_amount=2000000
)

print(f"Optimal Trading Path: {best_path}")
print(f"Path Detail: {' → '.join(best_path_detail)}")
print(f"Starting Amount: {format_amount(2000000)} SeaShells")
print(f"Final Amount: {format_amount(best_final_amount)} {best_path_detail[-1]}")
print(f"Profit Percentage: {best_profit:.2f}%")

# Analyze all possible cycles that start and end with SeaShells
print("\nAnalyzing all profitable cycles (starting and ending with SeaShells):")
all_cycles = []
seashells_idx = 3  # Index of SeaShells

# Find all possible cycles of length 2 to 4 that start and end with SeaShells
for path_length in range(2, 5):
    if path_length == 2:
        # Special case for length 2 (just one middle asset)
        for middle_asset in range(4):
            if middle_asset != seashells_idx:
                path = (seashells_idx, middle_asset, seashells_idx)
                test_amount = 1000
                final_amount, profit_percentage, path_names = calculate_profit(path, test_amount)
                
                if profit_percentage > 0:
                    all_cycles.append((path, profit_percentage, path_names))
    else:
        # Get all possible combinations of middle assets
        for middle_assets in permutations([i for i in range(4) if i != seashells_idx], path_length - 2):
            path = (seashells_idx,) + middle_assets + (seashells_idx,)
            test_amount = 1000
            final_amount, profit_percentage, path_names = calculate_profit(path, test_amount)
            
            if profit_percentage > 0:
                all_cycles.append((path, profit_percentage, path_names))

# Sort cycles by profit percentage (descending)
all_cycles.sort(key=lambda x: x[1], reverse=True)

# Print top 5 most profitable cycles
print("\nTop 5 most profitable cycles:")
for i, (path, profit, path_names) in enumerate(all_cycles[:5]):
    print(f"{i+1}. {' → '.join(path_names)}: {profit:.2f}%")

# Simulate the optimal trading strategy
print("\nSimulating the optimal trading strategy:")
current_amount = 2000000
print(f"Starting with {format_amount(current_amount)} SeaShells")

if best_profit > 0:
    # If profitable, follow the optimal path
    for i in range(len(best_path)-1):
        from_idx = best_path[i]
        to_idx = best_path[i+1]
        from_asset = assets[from_idx]
        to_asset = assets[to_idx]
        exchange_rate = rates[from_idx][to_idx]
        new_amount = current_amount * exchange_rate
        
        print(f"Convert {format_amount(current_amount)} {from_asset} → {format_amount(new_amount)} {to_asset}")
        current_amount = new_amount

    print(f"Final amount: {format_amount(current_amount)} {assets[best_path[-1]]}")
    print(f"Total profit: {current_amount - 2000000:,.2f} ({best_profit:.2f}%)")