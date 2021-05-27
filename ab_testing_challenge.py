from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory
from pywebio.input import *
from pywebio.output import *
import argparse
from pywebio import start_server
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

np.random.seed(1)

app = Flask(__name__)


# defining the bandit set up
class ABBandit:
    def __init__(self):
        self.reward_per_win = 0.40
        self.bandits = ['A', 'B', 'C']
        self.played = {'A': 0,
                       'B': 0,
                       'C': 0}
        self.wins = {'A': 0,
                     'B': 0,
                     'C': 0}
        self.actual_win_rate = {'A': 0.3,
                                'B': 0.2,
                                'C': 0.35}
        self.observed_win_rate = {'A': 0.0,
                                  'B': 0.0,
                                  'C': 0.0}
        self.money_won = {'A': 0.0,
                          'B': 0.0,
                          'C': 0.0}
        self.overall_played = 0
        self.overall_wins = 0
        self.overall_winrate = 0
        self.overall_money_won = 0
        self.games_left = 1000

    def pull_arm(self, bandit):
        if self.games_left > 0:
            random_chance = np.random.rand()
            self.played[bandit] += 1
            self.overall_played += 1
            if random_chance < self.actual_win_rate[bandit]:
                self.wins[bandit] += 1
                self.overall_wins += 1
            else:
                pass
            self.observed_win_rate[bandit] = self.wins[bandit] / self.played[bandit]
            self.money_won[bandit] = self.wins[bandit] * self.reward_per_win
            self.overall_winrate = self.overall_wins / self.overall_played
            self.overall_money_won = self.overall_wins * self.reward_per_win
            self.games_left = 1000 - self.overall_played
        else:
            pass
        # update df_overview
        core_data = {'Bandits': list(bandit_challenge.bandits),
                     'Played': list(bandit_challenge.played.values()),
                     'Wins': list(bandit_challenge.wins.values()),
                     'Winrate': list(bandit_challenge.observed_win_rate.values())
                     }
        df_overview = pd.DataFrame(core_data, columns=['Bandits', 'Played', 'Wins', 'Winrate'])
        return df_overview


def generate_bar_charts(df, x_axis, y_axis, title):
    fig = px.bar(df, x=x_axis, y=y_axis,
                 color=y_axis,
                 height=400)
    fig.update_layout(title_text=title)
    html = fig.to_html(include_plotlyjs="require", full_html=False)
    return put_html(html).send()


# instantiate bandits
bandit_challenge = ABBandit()
run_web_page = True


def run_experiment():
    while run_web_page:
        add_more = True

        while add_more:
            add_more = actions(label="Which bandit do you chose?",
                               buttons=[{'label': 'A', 'value': 'A'},
                                        {'label': 'B', 'value': 'B'},
                                        {'label': 'C', 'value': 'C'}])

            df_overview = bandit_challenge.pull_arm(add_more)
            fig = go.Figure(go.Indicator(
                mode="number+gauge+delta",
                gauge={'shape': "bullet"},
                delta={'reference': 1000},
                value=df_overview['Played'].sum(),
                domain={'x': [0.1, 1], 'y': [0.2, 0.9]},
                title={'text': "No. played"}))
            html = fig.to_html(include_plotlyjs="require", full_html=False)
            put_html(html).send()

            generate_bar_charts(df_overview, 'Bandits', 'Played', 'Rounds played')
            generate_bar_charts(df_overview, 'Bandits', 'Winrate', 'Overview of winrates')

            continue_button = actions(label="Do you want to continue",
                                      buttons=[{'label': 'Yes', 'value': True},
                                               {'label': 'No', 'value': False}])
            if continue_button:
                clear()


app.add_url_rule('/tool', 'webio_view', webio_view(run_experiment),
                 methods=['GET', 'POST', 'OPTIONS'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()

    start_server(run_experiment, port=args.port)
