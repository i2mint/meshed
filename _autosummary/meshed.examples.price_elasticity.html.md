# meshed.examples.price_elasticity

price elasticity relates price to revenue, expense, and profit
: ┌─────────┐
  │  base   │
  └─────────┘
  <br/>
  > │
  > │
  > ▼

┌────────────┐     ┌─────────────────────────┐
│ elasticity │ ──▶ │          sold           │ ─┐
└────────────┘     └─────────────────────────┘  │

> │               ▲          │
> │               │          │
> ▼               │          │

┌────────────┐     ┌─────────┐     ┌─────────┐  │
│    cost    │ ──▶ │ expense │     │  price  │  │
└────────────┘     └─────────┘     └─────────┘  │

> > │               │          │
> > │               │          │
> > ▼               ▼          │

> ┌─────────┐     ┌─────────┐  │
> │ profit  │ ◀── │ revenue │ ◀┘
> └─────────┘     └─────────┘

### Functions

| `expense`(cost, sold)             |    |
|-----------------------------------|----|
| `profit`(revenue, expense)        |    |
| `revenue`(price, sold)            |    |
| `sold`(price, elasticity[, base]) |    |
