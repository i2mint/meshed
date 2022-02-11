"""Simple model relating vaccination to death toll, involving exposure and infection rate

                        ┌──────────────────────┐
                        │   death_vax_factor   │
                        └──────────────────────┘
                          │
                          ▼
┌─────────────────┐     ┌──────────────────────────────────┐
│ die_if_infected │ ──▶ │               die                │
└─────────────────┘     └──────────────────────────────────┘
                          │                       ▲    ▲
                          ▼                       │    │
┌─────────────────┐     ┌──────────────────────┐  │  ┌─────┐
│   population    │ ──▶ │      death_toll      │  │  │ vax │
└─────────────────┘     └──────────────────────┘  │  └─────┘
                                                  │    │
                        ┌──────────────────────┐  │    │
                        │ infection_vax_factor │  │    │
                        └──────────────────────┘  │    │
                          │                       │    │
                          ▼                       │    │
                        ┌──────────────────────┐  │    │
                     ┌▶ │       infected       │ ─┘    │
                     │  └──────────────────────┘       │
                     │    ▲                            │
                     │    └────────────────────────────┘
                     │  ┌──────────────────────┐
                     │  │       exposed        │
                     │  └──────────────────────┘
                     │    │
                     │    ▼
                     │  ┌──────────────────────┐
                     └─ │          r           │
                        └──────────────────────┘
                          ▲
                          │
                        ┌──────────────────────┐
                        │   infect_if_expose   │
                        └──────────────────────┘
"""

DFLT_VAX = 0.5


def _factor(vax, vax_factor):
    assert 0 <= vax <= 1, 'vax should be between 0 and 1: Was {vax}'
    return vax * vax_factor + (1 - vax)


def r(exposed: float = 6, infect_if_expose: float = 1 / 5):
    return exposed * infect_if_expose


def infected(r: float = 1.2, vax: float = DFLT_VAX, infection_vax_factor: float = 0.15):
    return r * _factor(vax, infection_vax_factor)


def die(
    infected: float,
    die_if_infected: float = 0.05,
    vax: float = DFLT_VAX,
    death_vax_factor: float = 0.05,
):
    return infected * die_if_infected * _factor(vax, death_vax_factor)


def death_toll(die: float, population: int = 1e6):
    return int(die * population)
