# meshed.examples.price_elasticity

Price elasticity relates price to revenue, expense, and profit.

```text
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
```

### Functions

| `expense`(cost, sold)             |    |
|-----------------------------------|----|
| `profit`(revenue, expense)        |    |
| `revenue`(price, sold)            |    |
| `sold`(price, elasticity[, base]) |    |
