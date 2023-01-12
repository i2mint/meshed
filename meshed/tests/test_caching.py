"""Tests for caching module"""


def test_lazy_props():
    # Note: LazyProps isn't used at the time of writing this (2022-05-20) so if
    # test fails can (maybe) remove.
    from meshed.caching import LazyProps
    from i2 import LiteralVal

    # TODO: Why doesn't it work with dataclasses?
    # from dataclasses import dataclass
    # @dataclass
    class Funnel(LazyProps):
        #     impressions: int = 1000,
        #     cost_per_impression: float = 0.001,
        #     click_per_impression: float = 0.02,  # aka click through rate
        #     sales_per_click: float = 0.05,
        #     revenue_per_sale: float = 100.00

        def __init__(
            self,
            impressions: int = 1000,
            cost_per_impression: float = 0.001,
            click_per_impression: float = 0.02,  # aka click through rate
            sales_per_click: float = 0.05,
            revenue_per_sale: float = 100.00,
        ):  # aka average basket value
            self.impressions = impressions
            self.cost_per_impression = cost_per_impression
            self.click_per_impression = click_per_impression
            self.sales_per_click = sales_per_click
            self.revenue_per_sale = revenue_per_sale

        def cost(self):
            return self.impressions * self.cost_per_impression

        def clicks(self):
            return self.impressions * self.click_per_impression

        def sales(self):
            return self.clicks * self.sales_per_click

        def revenue(self):
            return self.sales * self.revenue_per_sale

        def profit(self):
            return self.revenue - self.cost

        @LiteralVal  # Meaning "leave this attribute as is (i.e. don't make it a lazy prop)"
        def leave_this_alone(self, a, b):
            return a + b

    f = Funnel(impressions=100, sales_per_click=0.15)
    assert f.revenue == 30.0
    assert f.leave_this_alone(1, 2) == 3

    f = Funnel(click_per_impression=0.04)
    assert (f.revenue, f.cost, f.profit) == (200.0, 1.0, 199.0)
