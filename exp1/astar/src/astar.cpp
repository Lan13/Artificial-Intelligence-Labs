#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<queue>
#include<set>
#include<algorithm>

using namespace std;

// A* algorithm state
class State {
public:
    State(vector<vector<bool>> maze, int g, int cnt, State *parent, int x, int y, int s) : 
    maze(maze), g(g), cnt(cnt), parent(parent), x(x), y(y), s(s) {
        h = cnt;
        f = g + h;
    }

    // check if the current state is the goal state
    bool is_goal() {
        return cnt == 0;
    }

    bool operator<(const State &other) const {
        return f < other.f;
    }

    bool operator==(const State &other) const {
        return maze == other.maze;
    }

    // get the successors of the current state
    vector<State *> get_successors() {
        vector<State *> successors;
        int n = maze.size();
        // for each successor
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // solution 1
                if (i - 1 >= 0 && j + 1 < n) {
                    if (maze[i][j] == 1 || maze[i - 1][j] == 1 || maze[i][j + 1] == 1) {
                        vector<vector<bool>> maze1 = maze;
                        maze1[i][j] = maze1[i][j] ^ 1;
                        maze1[i - 1][j] = maze1[i - 1][j] ^ 1;
                        maze1[i][j + 1] = maze1[i][j + 1] ^ 1;
                        int curr_cnt = cnt - maze[i][j] - maze[i - 1][j] - maze[i][j + 1];
                        curr_cnt = curr_cnt + maze1[i][j] + maze1[i - 1][j] + maze1[i][j + 1];
                        State *new_state = new State(maze1, g + 3, curr_cnt, this, i, j, 1);
                        successors.push_back(new_state);
                    }
                }
                // solution 2
                if (i - 1 >= 0 && j - 1 >= 0) {
                    if (maze[i][j] == 1 || maze[i - 1][j] == 1 || maze[i][j - 1] == 1) {
                        vector<vector<bool>> maze2 = maze;
                        maze2[i][j] = maze2[i][j] ^ 1;
                        maze2[i - 1][j] = maze2[i - 1][j] ^ 1;
                        maze2[i][j - 1] = maze2[i][j - 1] ^ 1;
                        int curr_cnt = cnt - maze[i][j] - maze[i - 1][j] - maze[i][j - 1];
                        curr_cnt = curr_cnt + maze2[i][j] + maze2[i - 1][j] + maze2[i][j - 1];
                        State *new_state = new State(maze2, g + 3, curr_cnt, this, i, j, 2);
                        successors.push_back(new_state);
                    }
                }
                // solution 3
                if (i + 1 < n && j - 1 >= 0) {
                    if (maze[i][j] == 1 || maze[i + 1][j] == 1 || maze[i][j - 1] == 1) {
                        vector<vector<bool>> maze3 = maze;
                        maze3[i][j] = maze3[i][j] ^ 1;
                        maze3[i + 1][j] = maze3[i + 1][j] ^ 1;
                        maze3[i][j - 1] = maze3[i][j - 1] ^ 1;
                        int curr_cnt = cnt - maze[i][j] - maze[i + 1][j] - maze[i][j - 1];
                        curr_cnt = curr_cnt + maze3[i][j] + maze3[i + 1][j] + maze3[i][j - 1];
                        State *new_state = new State(maze3, g + 3, curr_cnt, this, i, j, 3);
                        successors.push_back(new_state);
                    }
                }
                // solution 4
                if (i + 1 < n && j + 1 < n) {
                    if (maze[i][j] == 1 || maze[i + 1][j] == 1 || maze[i][j + 1] == 1) {
                        vector<vector<bool>> maze4 = maze;
                        maze4[i][j] = maze4[i][j] ^ 1;
                        maze4[i + 1][j] = maze4[i + 1][j] ^ 1;
                        maze4[i][j + 1] = maze4[i][j + 1] ^ 1;
                        int curr_cnt = cnt - maze[i][j] - maze[i + 1][j] - maze[i][j + 1];
                        curr_cnt = curr_cnt + maze4[i][j] + maze4[i + 1][j] + maze4[i][j + 1];
                        State *new_state = new State(maze4, g + 3, curr_cnt, this, i, j, 4);
                        successors.push_back(new_state);
                    }
                }
            }
        }
        return successors;
    }

public:
    int x, y, s; // position and solution
    int g, f, h, cnt; // cost
    State *parent; // parent state
    vector<vector<bool>> maze; // maze
};

static bool compare(const State *a, const State *b) {
    return a->f < b->f;
};

bool solve(State *start, string filename) {
    multiset<State *, decltype(compare)*> open_list{compare}; // open list
    open_list.insert(start); // add the start state to the open list
    while (!open_list.empty()) {
        // get the state with the smallest f value
        auto current = *open_list.begin();
        open_list.erase(open_list.begin());

        if (current->is_goal()) { // if the current state is the goal state
            // get path
            vector<State *> path;
            while (current->parent != nullptr) {
                path.push_back(current);
                current = current->parent;
            }
            reverse(path.begin(), path.end());
            ofstream fout(filename);
            if (!fout.is_open()) {
                cout << "Error opening output file" << endl;
                return false;
            }
            // write path to file
            fout << path.size() << endl;
            for (auto state : path) {
                fout << state->x << "," << state->y << "," << state->s << endl;
            }
            cout << "final step: " << path.size() << endl;
            return true;
        }

        // generate successors
        vector<State *> successors = current->get_successors();
        // for each successor
        for (auto successor : successors) {
            // if the successor is not in the open list
            open_list.insert(successor);

            // Simplified Memory Bounded A*
            if (open_list.size() > 10000) {
                for (int k = 0; k < 200; k++) {
                    auto it = open_list.end();
                    it--;
                    open_list.erase(it);
                    delete *it;
                }
            } 
        }
    }

    ofstream fout(filename);
    if (!fout.is_open()) {
        cout << "Error opening output file" << endl;
        return false;
    }
    fout << "No valid solution." << endl;
    return false;
}


int main() {
    // read maze from file
    string input_filename = "../input/input9.txt";
    string output_filename = "../output/output9.txt";
    ifstream fin(input_filename);
    if (!fin.is_open()) {
        cout << "Error opening input file" << endl;
        return 0;
    }
    int n, cnt = 0;
    bool num;
    fin >> n;
    vector<vector<bool>> maze(n, vector<bool>(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fin >> num;
            maze[i][j] = num;
            cnt += num;
        }
    }
    fin.close();
    // solve the maze
    State *start = new State(maze, 0, cnt, nullptr, -1, -1, -1);
    solve(start, output_filename);
    return 0;
}