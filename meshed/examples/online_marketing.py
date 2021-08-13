"""Online marketing funnel: impressions and clicks to sales and profit.

                             ┌──────────────────────┐
                             │ click_per_impression │
                             └──────────────────────┘
                               │
                               ▼
     ┌─────────────────┐     ┌──────────────────────┐
     │   impressions   │ ──▶ │        clicks        │
     └─────────────────┘     └──────────────────────┘
  ┌────┘                       │
  │                            ▼
  │  ┌─────────────────┐     ┌──────────────────────┐
  │  │ sales_per_click │ ──▶ │        sales         │
  │  └─────────────────┘     └──────────────────────┘
  │                            │
  │                            ▼
  │                          ┌──────────────────────┐     ┌──────────────────┐
  │                          │       revenue        │ ◀── │ revenue_per_sale │
  │                          └──────────────────────┘     └──────────────────┘
  │                            │
  │                            ▼
  │                          ┌──────────────────────┐
  │                          │        profit        │ ◀┐
  │                          └──────────────────────┘  │
  │                          ┌──────────────────────┐  │
  │                          │ cost_per_impression  │  │
  │                          └──────────────────────┘  │
  │                            │                       │
  │                            ▼                       │
  │                          ┌──────────────────────┐  │
  └────────────────────────▶ │         cost         │ ─┘
                             └──────────────────────┘

"""


def cost(impressions, cost_per_impression):
    return impressions * cost_per_impression


def clicks(impressions, click_per_impression):
    return impressions * click_per_impression


def sales(clicks, sales_per_click):
    return clicks * sales_per_click


def revenue(sales, revenue_per_sale):
    return sales * revenue_per_sale


def profit(revenue, cost):
    return revenue - cost


funcs = (cost, clicks, sales, revenue, profit)
