#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<queue>
#include<tuple>
#include<algorithm>
#include<math.h>

using namespace std;

// CSP algorithm state
class ShiftSchedule {
public:

    ShiftSchedule(int staff_num, int days_num, int shifts_num, vector<vector<int>> requests) {
        this->staffs_num = staff_num;
        this->days_num = days_num;
        this->shifts_num = shifts_num;
        this->requests = requests;
        this->schedule = vector<vector<int>>(days_num, vector<int>(shifts_num, -1));
        this->assigned_counts = vector<int>(staff_num, 0);
        this->valid = false;
        this->fulfilled_requests_num = 0;
    }

    void assign(int staff_id, int day_id, int shift_id) {
        schedule[day_id][shift_id] = staff_id;
        assigned_counts[staff_id]++;
    }

    void unassign(int staff_id, int day_id, int shift_id) {
        schedule[day_id][shift_id] = -1;
        assigned_counts[staff_id]--;
    }

    bool check(int staff_id, int day_id, int shift_id) {
        // check if the staff is already assigned
        if (schedule[day_id][shift_id] != -1) {
            return false;
        }
        // check if the staff member is assigned to the previous shift
        if ((shift_id > 0) && (schedule[day_id][shift_id - 1] == staff_id)) {
            return false;
        }
        // check if the staff member is assigned to the next shift
        if ((shift_id < shifts_num - 1) && (schedule[day_id][shift_id + 1] == staff_id)) {
            return false;
        }
        // check if the staff member is assigned to the last shift of the previous day
        if ((day_id > 0) && (shift_id == 0) && (schedule[day_id - 1][shifts_num - 1] == staff_id)) {
            return false;
        }
        // check if the staff member is assigned to the first shift of the next day
        if ((day_id < days_num - 1) && (shift_id == shifts_num - 1) && (schedule[day_id + 1][0] == staff_id)) {
            return false;
        }
        return true;
    }

    // pruning the search tree
    bool prune(int staff_id, int day_id, int shift_id) const{
        // if the staff member is already assigned enough shifts, do not prune the search tree
        if (assigned_counts[staff_id] >= (days_num * shifts_num) / staffs_num) {
            return false;
        }
        // only prune the search tree when the staff member does not assign enough shifts
        // and there are no requests for the rest shifts
        for (int d = day_id; d < days_num; d++) {
            for (int s = shift_id; s < shifts_num; s++) {
                // if the staff member is able to assigned in the following shifts, do not prune the search tree
                if (requests[staff_id][d * shifts_num + s] == 1) {
                    return false;
                }
            } 
        }
        // if any staff member is unable to assigned in the following shifts, prune the search tree
        return true;
    }

    int count_fulfilled_requests() const{
        int count = 0;
        for (int d = 0; d < days_num; d++) {
            for (int s = 0; s < shifts_num; s++) {
                for (int n = 0; n < staffs_num; n++) {
                    if (requests[n][d * shifts_num + s] == 1 && schedule[d][s] == n) {
                        count++;
                    }
                }
            }
        }
        return count;
    }

    vector<int> get_unassigned_staff_ids() const {
        vector<int> unassigned_staff_ids;
        for (int i = 0; i < staffs_num; i++) {
            unassigned_staff_ids.push_back(i);
        }
        stable_sort(unassigned_staff_ids.begin(), unassigned_staff_ids.end(), 
            [&](int pos1, int pos2) { return (assigned_counts[pos1] < assigned_counts[pos2]); });
        return unassigned_staff_ids;
    }
public:
    int staffs_num;
    int days_num;
    int shifts_num;
    vector<vector<int>> requests;
    vector<vector<int>> schedule;
    vector<int> assigned_counts;
    bool valid;
    int fulfilled_requests_num;
};

// backtracking algorithm
ShiftSchedule backtrack(ShiftSchedule schedule) {
    // check if all shifts are assigned
    bool all_assigned = true;
    for (int d = 0; d < schedule.days_num; d++) {
        for (int s = 0; s < schedule.shifts_num; s++) {
            if (schedule.schedule[d][s] == -1) {
                all_assigned = false;
                break;
            }
        }
        if (!all_assigned) {
            break;
        }
    }

    // if all shifts are assigned, check if requests are fulfilled
    if (all_assigned) {
        schedule.fulfilled_requests_num = schedule.count_fulfilled_requests();
        schedule.valid = true;
        return schedule;
    }

    // select the next shift to assign
    vector<int> unassigned_staff_ids = schedule.get_unassigned_staff_ids();
    priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>,
                   greater<pair<int, pair<int, int>>>> shift_candidates;
    
    for (int d = 0; d < schedule.days_num; d++) {
        for (int s = 0; s < schedule.shifts_num; s++) {
            // use MRV heuristic to select the next shift to assign
            int count = 0;
            for (int staff_id : unassigned_staff_ids) {
                if (schedule.check(staff_id, d, s)) {
                    count++;
                }
            }
            if (count > 0) {
                shift_candidates.push(make_pair(count, make_pair(d, s)));
            }
            // find an unassigned shift that is unable to assigned for 0 candidates
            if (count == 0 && schedule.schedule[d][s] == -1) {
                schedule.valid = false;
                return schedule;
            }
        }
    }
    
    ShiftSchedule best_schedule(schedule);
    // assign the selected shift to a staff member
    while (!shift_candidates.empty()) {
        auto shift_candidate = shift_candidates.top();
        shift_candidates.pop();
        int day_id = (shift_candidate.second).first;
        int shift_id = (shift_candidate.second).second;
        bool assigned = false;
        double temp = ((double)(schedule.days_num * schedule.shifts_num) / (double)schedule.staffs_num);
        int max_assigned_counts = ceil(temp);
        for (int staff_id : unassigned_staff_ids) {
            // prune the search tree
            if (schedule.prune(staff_id, day_id, shift_id)) {
                break;
            }
            // check if staff can be assigned
            if (schedule.check(staff_id, day_id, shift_id) 
                && schedule.requests[staff_id][day_id * schedule.shifts_num + shift_id] == 1
                && schedule.assigned_counts[staff_id] < max_assigned_counts) {
                
                assigned = true;
                schedule.assign(staff_id, day_id, shift_id);
                
                auto result = backtrack(schedule);
                // check if result is valid
                if (result.valid && result.fulfilled_requests_num > best_schedule.fulfilled_requests_num) {
                    bool flag = true;
                    // check if all staff members are assigned enough shifts
                    for (int i = 0; i < result.staffs_num; i++) {
                        if (result.assigned_counts[i] < (result.days_num * result.shifts_num) / result.staffs_num) {
                            cout << "Error: Staff " << i + 1 << " is not assigned enough shifts" << endl;
                            flag = false;
                            break;
                        }
                    }
                    if (flag) {
                        best_schedule = result;
                        return best_schedule;
                    }
                    // return best_schedule;
                }
                schedule.unassign(staff_id, day_id, shift_id);
            }
        }
        // if not assigned, choose any staff member to assign
        if (!assigned) {
            for (int staff_id : unassigned_staff_ids) {
                // check if staff can be assigned
                if (schedule.check(staff_id, day_id, shift_id)) {
                    schedule.assign(staff_id, day_id, shift_id);
                    auto result = backtrack(schedule);
                    // check if result is valid
                    if (result.valid && result.fulfilled_requests_num >= best_schedule.fulfilled_requests_num) {
                        bool flag = true;
                        // check if all staff members are assigned enough shifts
                        for (int i = 0; i < result.staffs_num; i++) {
                            if (result.assigned_counts[i] < (result.days_num * result.shifts_num) / result.staffs_num) {
                                cout << "Error: Staff " << i + 1 << " is not assigned enough shifts" << endl;
                                flag = false;
                                break;
                            }
                        }
                        if (flag) {
                            best_schedule = result;
                            return best_schedule;
                        }
                        // return best_schedule;
                    }
                    schedule.unassign(staff_id, day_id, shift_id);
                }
            }
        }
    }
    return best_schedule;
}

// using minimum remaining values and backtracking to solve the schedule
bool solve(int staffs_num, int days_num, int shifts_num, vector<vector<int>> requests, string filename) {
    // start state
    ShiftSchedule schedule(staffs_num, days_num, shifts_num, requests);
    auto best_schedule = backtrack(schedule);
    if (best_schedule.valid) {
        // check if all staff members are assigned enough shifts
        for (int i = 0; i < staffs_num; i++) {
            if (best_schedule.assigned_counts[i] < (days_num * shifts_num) / staffs_num) {
                cout << "Error: Staff " << i + 1 << " is not assigned enough shifts" << endl;
                return false;
            }
        }
        ofstream fout(filename);
        if (!fout.is_open()) {
            cout << "Error opening output file" << endl;
            return false;
        }
        // print the shift schedule and the number of fulfilled requests
        for (int d = 0; d < best_schedule.days_num; d++) {
            for (int s = 0; s < best_schedule.shifts_num; s++) {
                if (s == best_schedule.shifts_num - 1) {
                    fout << best_schedule.schedule[d][s] + 1;
                    break;
                }
                fout << best_schedule.schedule[d][s] + 1 << ",";
            }
            fout << endl;
        }
        fout << best_schedule.fulfilled_requests_num;
        cout << "Fulfilled requests: " << best_schedule.fulfilled_requests_num << endl;
        fout.close();
        return true;
    }
    else {
        ofstream fout(filename);
        if (!fout.is_open()) {
            cout << "Error opening output file" << endl;
            return false;
        }
        fout << "No valid schedule found." << endl;
        fout.close();
        return false;
    }
}

int main() {
    // read schedule from file
    string input_filename = "../input/input9.txt";
    string output_filename = "../output/output9.txt";
    ifstream fin(input_filename);
    if (!fin.is_open()) {
        cout << "Error opening input file" << endl;
        return 0;
    }
    int staffs_num, days_num, shifts_num;
    char comma;
    fin >> staffs_num >> comma >> days_num >> comma >> shifts_num;
    
    vector<vector<int>> requests(staffs_num, vector<int>(days_num * shifts_num));
    for (int i = 0; i < staffs_num; i++) {
        for (int j = 0; j < days_num; j++) {
            for (int k = 0; k < shifts_num; k++) {
                if (k == shifts_num - 1) {
                    fin >> requests[i][j * shifts_num + k];
                } else {
                    fin >> requests[i][j * shifts_num + k] >> comma;
                }
            }
        }
    }
    fin.close();

    solve(staffs_num, days_num, shifts_num, requests, output_filename);
    return 0;
}