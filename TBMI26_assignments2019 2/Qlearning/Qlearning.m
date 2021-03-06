%% Initialization
%  Initialize the world, Q-table, and hyperparameters
world = 12;
gwinit(world);
initial_state = gwstate;
Q = rand(initial_state.ysize, initial_state.xsize, 4);
Q(initial_state.ysize, :, 1) = -inf;
Q(1, :, 2) = -inf;
Q(:, initial_state.xsize, 3) = -inf;
Q(:, 1, 4) = -inf;
learning_rate = 0.25;
discount_factor = 0.9;
initial_epsilon = 0.9;
nr_of_episodes = 2000;
gwdraw

%% Training loop
%  Train the agent using the Q-learning algorithm.
for episode=1:nr_of_episodes
    gwinit(world)
    state = gwstate;
    epsilon = initial_epsilon*(1 - 0.95*episode/nr_of_episodes);
    while state.isterminal~=1
        action = chooseaction(Q, state.pos(1), state.pos(2), [1 2 3 4], [1 1 1 1], epsilon);
        new_state = gwaction(action);
        if new_state.isvalid
            Q(state.pos(1), state.pos(2), action) = (1 - learning_rate)*Q(state.pos(1), state.pos(2), action) + learning_rate*(new_state.feedback + discount_factor*max(Q(new_state.pos(1), new_state.pos(2), :)));
            state = new_state;
        %else %Penalty for avoiding pillars
        %    Q(state.pos(1), state.pos(2), action) = (1 - learning_rate)*Q(state.pos(1), state.pos(2), action) + learning_rate*(-11 + discount_factor*max(Q(new_state.pos(1), new_state.pos(2), :)));
        end
    end
    if state.isterminal==1
        Q(state.pos(1), state.pos(2), :) = 0;
    end
end

%% Test loop
%  Test the agent (subjectively) by letting it use the optimal policy
%  to traverse the gridworld. Do not update the Q-table when testing.
%  Also, you should not explore when testing, i.e. epsilon=0, always pick
%  the optimal action.
optimal_policy = gwgetpolicy(Q);
gwinit(world);
state = gwstate;
while state.isterminal ~= 1
    action = optimal_policy(state.pos(1), state.pos(2));
    state = gwaction(action);
    gwdraw()
end
gwdraw(Q)

%% Plotting policy and V-function
optimal_policy = gwgetpolicy(Q);
imagesc(max(Q, [], 3))
global GWTERM;
global GWFEED;
for x = 1:initial_state.xsize
  for y = 1:initial_state.ysize
    if ~GWTERM(y,x) && ~isnan(GWFEED(y,x))
        gwplotarrow([y x], optimal_policy(y, x));
    end
  end
end
title(sprintf('V-function and policy, World %i', world))

%% Random Tests
%gwinit(world)
state = gwaction(1)
disp(state.pos)
gwdraw()

%% More testing
%imagesc(Q(:,:,4))
gwdraw(Q)