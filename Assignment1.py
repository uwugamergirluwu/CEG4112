import matplotlib.pyplot as plt
from icecream import ic

'''
requirements: 8 gpus

Each On-Prem Server: 32-core server, with 256 GB RAM and 2 TB SSD.
Cloud: AWS p4d.24xlarge instances (8 GPUs, 96 vCPUs, 1.1 TB RAM, and 8 TB NVMe SSD).

8hrs/day, 22days/month, 176hrs/month
5TB data rate with cloud
AWS p4d.24xlarge: $32.77/hour (locked in for 4-year term). Data transfer 5 TB/month at $0.09/GB. 

On-prem Hardware: Each server (4 GPUs, 32-core server, SSD, networking): $65,800. Power, cooling, IT staff: $5,000/year per server.

Show the TCO calculation leading to 4-Year TCO On-Prem of ~$171,600 vs 4-Year TCO Cloud of ~$298,500
'''
# Constants
on_prem_server_cost = 65_800
on_prem_yearly_maintenance_per_server = 5_000
monthly_usage_hours = 176
data_transfer_in_GB = 5000
aws_hourly_rate = 32.77
aws_monthly_data_rate_per_GB = 0.09

#Calculation of monthly cost
aws_monthly_cost = aws_hourly_rate * monthly_usage_hours + aws_monthly_data_rate_per_GB * data_transfer_in_GB
#since we need 8 GPUs, two on-prem servers are needed
initial_server_cost = on_prem_server_cost * 2
monthly_server_maintenance = (on_prem_yearly_maintenance_per_server/12) * 2

#Calculation of costs over time

months=0
total_aws_cost=0
total_on_prem_cost = initial_server_cost
aws_costs = []
on_prem_costs = []
month_list = []

while months < 48:
    months+=1
    total_aws_cost += aws_monthly_cost
    total_on_prem_cost += monthly_server_maintenance
    aws_costs.append(total_aws_cost)
    on_prem_costs.append(total_on_prem_cost)
    month_list.append(months)

break_even_point_months = next((i for i, cost in enumerate(aws_costs) if cost >= on_prem_costs[i]), None)

_=ic(break_even_point_months, total_aws_cost, total_on_prem_cost)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(month_list, aws_costs, label='AWS')
plt.plot(month_list, on_prem_costs, label='On-prem')
plt.axvline(x=break_even_point_months, color='r', linestyle='--', label=f'Break-even point ({break_even_point_months} months)')
plt.xlabel('Months')
plt.ylabel('Total Cost ($)')
plt.title('Cost Comparison: 5TB of outbound traffic monthly on AWS vs. On-Premise over four years')
plt.legend()
plt.grid(True)
plt.show()