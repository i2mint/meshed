# meshed.examples.online_marketing

Online marketing funnel: impressions and clicks to sales and profit.

> > ┌──────────────────────┐
> > │ click_per_impression │
> > └──────────────────────┘

> > > │
> > > ▼

> ┌─────────────────┐     ┌──────────────────────┐
> │   impressions   │ ──▶ │        clicks        │
> └─────────────────┘     └──────────────────────┘

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

> └──────────────────────┘

### Functions

| `clicks`(impressions, click_per_impression)   |    |
|-----------------------------------------------|----|
| `cost`(impressions, cost_per_impression)      |    |
| `profit`(revenue, cost)                       |    |
| `revenue`(sales, revenue_per_sale)            |    |
| `sales`(clicks, sales_per_click)              |    |
