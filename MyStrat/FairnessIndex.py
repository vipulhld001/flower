def jains_fairness_index(allocations):
    # allocations is a list containing the allocations for each user
    sum_allocations = sum(allocations)
    sum_squared_allocations = sum(x**2 for x in allocations)
    num_users = len(allocations)

    fairness_index = (sum_allocations**2) / (num_users * sum_squared_allocations)
    return round(fairness_index, 3)

# Example usage with three users
user_allocations = [30, 40, 25,45]  # Replace with actual allocations
fairness_index = jains_fairness_index(user_allocations)

print(f"Jain's Fairness Index: {fairness_index}")