"""price elasticity relates price to revenue, expense, and profit
                   ┌─────────┐
                   │  base   │
                   └─────────┘
                     │
                     │
                     ▼
┌────────────┐     ┌─────────────────────────┐
│ elasticity │ ──▶ │          sold           │ ─┐
└────────────┘     └─────────────────────────┘  │
                     │               ▲          │
                     │               │          │
                     ▼               │          │
┌────────────┐     ┌─────────┐     ┌─────────┐  │
│    cost    │ ──▶ │ expense │     │  price  │  │
└────────────┘     └─────────┘     └─────────┘  │
                     │               │          │
                     │               │          │
                     ▼               ▼          │
                   ┌─────────┐     ┌─────────┐  │
                   │ profit  │ ◀── │ revenue │ ◀┘
                   └─────────┘     └─────────┘

"""


def profit(revenue, expense):
    return revenue - expense


def revenue(price, sold):
    return price * sold


def expense(cost, sold):
    return cost * sold


def sold(price, elasticity, base=1e6):
    return base * price ** (1 - elasticity)


funcs = (profit, revenue, expense, sold)
