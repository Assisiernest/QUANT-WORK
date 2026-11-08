print("-----Trade Profit-Loss Calculator-------")
account_balance = float(input("Enter account balance($):"))
risk_percentage = float(input("Enter risk percentage(%)P:"))
trade_result_r = float(input("Enter trade result(r):"))

# Calculate amount risked
risk_amount=account_balance * (risk_percentage/100)
# profit loss calculation
profit_loss = risk_amount * trade_result_r
# Calculate New account Balance
New_account_Balance = account_balance + profit_loss

# Display Results
print(f"Risk amount:${risk_amount:}")
print(f"profit loss:${profit_loss:}")
print(f"New Account Balance:${New_account_Balance:}")

if profit_loss > 0:
    print("Profit Realised")
elif profit_loss < 0:
    print("Loss")
else:
    print("Breakeven")
