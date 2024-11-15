clc
clear all
clf

% Close any existing parallel pool to ensure a fresh start
%delete(gcp('nocreate'));

% Start a process-based pool with a limited number of workers
numWorkers = 12;
parpool('Processes', numWorkers);

A=[0 0;0 0];
B=[1 0;0 1];
C=eye(2,2);
D=zeros(2,2);

sys=ss(A,B,C,D);


Num_agen=4;


Ts=0.1;
sysd=c2d(sys,Ts);
Ad=sysd.A;
Bd=sysd.B;
T=50;
%% intial trajectory
count=1;
%X(:,1)=[0;0 ;  20;0 ; 20;20   ;  0;20 ];
X(:,1)=[0;0 ;  0;5 ; 0;10   ;  0;15 ];
for t=0:Ts:20
    if t<2
        u(:,count)=0*ones(2*Num_agen,1);
    else
        u(:,count)=0*ones(2*Num_agen,1);
    end
    for i=1:Num_agen
        X((2*i-1):(2*i),count+1)=Ad*X((2*i-1):(2*i),count)+Bd*u((2*i-1):(2*i),count);
    end
    count=count+1;
end

X0=X;
for i=1:Num_agen
    hold on
    plot(X((2*i-1),:),X((2*i),:),'.')
end
N=length(X(1,:));

hold on





%% Laplacian initial traj

X(:,1)=[0;0 ;  0;5 ; 0;15   ;  0;10 ];
%X(:,1)=[20;25 ;  20;20 ; 25;20   ;  25;25 ];
kp = 2.8;
kc = 0.5;
Laplacian = - ones(Num_agen) + Num_agen*eye(Num_agen);
%
%
final = [20;25 ;  20;20 ; 25;20   ;  25;25 ];
final = [20;25 ;  20;20 ; 25;20   ;  25;25 ]+5;
for ii=1:Num_agen
    for jj=1:2
        final_trans(ii,jj)=final(2*(ii-1)+jj,1);
    end
end
final_centeroid=[sum(final_trans(:,1))/Num_agen,sum(final_trans(:,2))/Num_agen];



V_des=zeros(2*Num_agen,N-1);
X_trans=zeros(Num_agen,2);
for i=1:N-1
    for ii=1:Num_agen
        for jj=1:2
            X_trans(ii,jj)=X(2*(ii-1)+jj,i);
        end
    end
    centroid = (ones(Num_agen,Num_agen) * reshape(X_trans,[4,2]))/4;
    centroid_diff = centroid-[final_centeroid(1) final_centeroid(2);final_centeroid(1) final_centeroid(2);final_centeroid(1) final_centeroid(2);final_centeroid(1) final_centeroid(2)];

    V_des_trans = Laplacian*(final_trans-X_trans);



    V_des_trans=kp*V_des_trans - centroid_diff*kc;
    for ii=1:Num_agen
        for jj=1:2
            if V_des_trans(ii,jj)>=2
                V_des_trans(ii,jj)=2;
            elseif V_des_trans(ii,jj)<=-2
                V_des_trans(ii,jj)=-2;
            end
        end
    end
    V_des(:,i)=reshape(V_des_trans',[8,1]);
    for j=1:Num_agen
        X((j-1)*2+1:(j-1)*2+2,i+1)=Ad*X((j-1)*2+1:(j-1)*2+2,i)+Bd*V_des_trans(j,:)';
    end
    hold on
    plot(X(1,i),X(2,i),'r.');
    plot(X(3,i),X(4,i),'g.');
    plot(X(5,i),X(6,i),'b.');
    plot(X(7,i),X(8,i),'c.');
    xlim([-5 30])
    ylim([-5 30])
    pause(0.01)

end

u=V_des;
X0=X;
u0=u;


%%
R_obs=2;
R_agent=1.5;

obs_center=[12,90; ...
    X(1:2,1)'; ...
    X(3:4,1)'; ...
    X(5:6,1)'; ...
    X(7:8,1)'];
R_plot=[2*R_obs-R_agent,R_agent,R_agent,R_agent,R_agent];
R=2*R_plot;


alpha=1;
obs_num=length(R);
r_default=0.5;
r=0.5;

lambda=10000;

i=1;
figure(1)
tol=0.001;



hold on

theta=linspace(0,2*pi,201);
for j=1:obs_num
    x_theta=R_plot(j)*cos(theta);
    y_theta=R_plot(j)*sin(theta);
    plot(obs_center(j,1)+x_theta,obs_center(j,2)+y_theta)
end




%[X_out,u_out]=traj_gen(Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian)



% f1=parfeval(@traj_gen,2,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
% 
% f2=parfeval(@traj_gen,2,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
% Wait for both tasks to complete and fetch the results
% x1 = fetchOutputs(f1);
% x2 = fetchOutputs(f2);
%%
% [Cost_1,X_out1]=traj_gen(2,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
% %%
% for i=1:N
% subplot(2,2,1)
% plotting(X_out1,obs_center,R_plot,i)
% 
% 
% end
%obs_checker(X,Num_agen,obs_center,R)

% for iteration=1:10
% 
%     if iteration==1
%         [Cost_1,X_out1]=traj_gen(1,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
%         X_out=[X_out1(3:4,:);X(3:4,:);X(5:6,:);X(7:8,:)];
%         [Cost_2,X_out2]=traj_gen(2,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
%         X_out=[X_out1(3:4,:);X_out2(3:4,:);X(5:6,:);X(7:8,:)];
%         [Cost_3,X_out3]=traj_gen(3,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
%         X_out=[X_out1(3:4,:);X_out2(3:4,:);X_out3(3:4,:);X(7:8,:)];
%         [Cost_4,X_out4]=traj_gen(4,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
%         X_out=[X_out1(3:4,:);X_out2(3:4,:);X_out3(3:4,:);X_out4(3:4,:)];
% 
%     else
%         [Cost_1,X_out1]=traj_gen(1,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
%         disp("Cost 1    " + Cost_1)
%         X_out=[X_out1(3:4,:);X_out(3:4,:);X_out(5:6,:);X_out(7:8,:)];
%         obs_checker(X_out,Num_agen,obs_center,R)
%         [Cost_2,X_out2]=traj_gen(2,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
%         disp("Cost 2    " + Cost_2)
%         X_out=[X_out(1:2,:);X_out2(3:4,:);X_out(5:6,:);X_out(7:8,:)];
%         obs_checker(X_out,Num_agen,obs_center,R)
%         [Cost_3,X_out3]=traj_gen(3,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
%         disp("Cost 3    " + Cost_3)
%         X_out=[X_out(1:2,:);X_out(3:4,:);X_out3(3:4,:);X_out(7:8,:)];
%         obs_checker(X_out,Num_agen,obs_center,R)
%         [Cost_4,X_out4]=traj_gen(4,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
%         disp("Cost 4    " + Cost_4)
%         X_out=[X_out(1:2,:);X_out(3:4,:);X_out3(5:6,:);X_out4(3:4,:)];
%         obs_checker(X_out,Num_agen,obs_center,R)
% 
% 
%     end
% 
% 
%     iteration
%     Cost=[Cost_1,Cost_2,Cost_3,Cost_4];
%     disp(Cost)
%     total_cost(iteration)=sum(Cost);
% 
%     for i=1:N
% 
%         obs_center=[obs_center(1,:); ...
%             X_out(1:2,i)'; ...
%             X_out(3:4,i)'; ...
%             X_out(5:6,i)'; ...
%             X_out(7:8,i)'];
%         for j=1:Num_agen
%             countk=1;
%             for k=1:obs_num
%                 if k==j+1
%                     continue
%                 end
%                 ss((countk-1)*(N)+i,j)=R(k)-norm(X((2*j-1):(2*j),i)-obs_center(k,:)',2);
%                 countk=countk+1;
%             end
%         end
% 
%     end
% 
%     ss_max=max(ss);
% 
%     if ss_max<=0
%         break;
%     end
% 
% end


%% Beginning of decentralized SCvx
total_cost=zeros(1,200);
for iteration=1:10

    if iteration==1
        f1=parfeval(@traj_gen,2,1,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
        f2=parfeval(@traj_gen,2,2,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
        f3=parfeval(@traj_gen,2,3,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
        f4=parfeval(@traj_gen,2,4,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
        [Cost_1,X_out1]=fetchOutputs(f1);
        [Cost_2,X_out2]=fetchOutputs(f2);
        [Cost_3,X_out3]=fetchOutputs(f3);
        [Cost_4,X_out4]=fetchOutputs(f4);
        X_out=[X_out1(3:4,:);X_out2(3:4,:);X_out3(3:4,:);X_out4(3:4,:)];
        % [X_out1,u_out1]=traj_gen(1,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
        % [X_out2,u_out2]=traj_gen(2,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
        % [X_out3,u_out3]=traj_gen(3,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
        % [X_out4,u_out4]=traj_gen(4,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian);
    else
        % [X_out1,u_out1]=traj_gen(1,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        % [X_out2,u_out2]=traj_gen(2,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        % [X_out3,u_out3]=traj_gen(3,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        % [X_out4,u_out4]=traj_gen(4,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        f1=parfeval(@traj_gen,2,1,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        f2=parfeval(@traj_gen,2,2,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        f3=parfeval(@traj_gen,2,3,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        f4=parfeval(@traj_gen,2,4,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        
        [Cost_1,X_out1]=fetchOutputs(f1);
        disp("Cost 1    " + Cost_1)
        X_out=[X_out1(3:4,:);X_out2(3:4,:);X_out3(3:4,:);X_out4(3:4,:)];
        obs_checker(X_out,Num_agen,obs_center,R)
        [Cost_2,X_out2]=fetchOutputs(f2);
        disp("Cost 2    " + Cost_2)
        X_out=[X_out1(3:4,:);X_out2(3:4,:);X_out3(3:4,:);X_out4(3:4,:)];
        obs_checker(X_out,Num_agen,obs_center,R)
        [Cost_3,X_out3]=fetchOutputs(f3);
        disp("Cost 3    " + Cost_3)
        X_out=[X_out1(3:4,:);X_out2(3:4,:);X_out3(3:4,:);X_out4(3:4,:)];
        obs_checker(X_out,Num_agen,obs_center,R)
        [Cost_4,X_out4]=fetchOutputs(f4);
        disp("Cost 4    " + Cost_4)
        X_out=[X_out1(3:4,:);X_out2(3:4,:);X_out3(3:4,:);X_out4(3:4,:)];
        obs_checker(X_out,Num_agen,obs_center,R)
        
        % f1=parfeval(@traj_gen,2,1,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        % f2=parfeval(@traj_gen,2,2,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        % f3=parfeval(@traj_gen,2,3,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        % f4=parfeval(@traj_gen,2,4,Ad,Bd,X_out,X0,u,R_plot,obs_center,final,Laplacian);
        % [Cost_1,X_out1]=fetchOutputs(f1);
        % [Cost_2,X_out2]=fetchOutputs(f2);
        % [Cost_3,X_out3]=fetchOutputs(f3);
        % [Cost_4,X_out4]=fetchOutputs(f4);
        % 
        % disp("Cost 1    " + Cost_1)
        % disp("Cost 2    " + Cost_2)
        % disp("Cost 3    " + Cost_3)
        % disp("Cost 4    " + Cost_4)
        % X_out=[X_out1(3:4,:);X_out2(3:4,:);X_out3(3:4,:);X_out4(3:4,:)];
    end
    


    iteration
    Cost=[Cost_1,Cost_2,Cost_3,Cost_4];
    disp(Cost)
    total_cost(iteration)=sum(Cost);
    
    for i=1:N

        obs_center=[obs_center(1,:); ...
            X_out(1:2,i)'; ...
            X_out(3:4,i)'; ...
            X_out(5:6,i)'; ...
            X_out(7:8,i)'];
        for j=1:Num_agen
            countk=1;
            for k=1:obs_num
                if k==j+1
                    continue
                end
                ss((countk-1)*(N)+i,j)=R(k)-norm(X((2*j-1):(2*j),i)-obs_center(k,:)',2);
                countk=countk+1;
            end
        end

    end

    ss_max=max(ss);

    if ss_max<=0
        break;
    end

end
%%






%%
figure(2)
for i=1:N
subplot(2,2,1)
plotting(X_out1,obs_center,R_plot,i)

subplot(2,2,2)
plotting(X_out2,obs_center,R_plot,i)

subplot(2,2,3)
plotting(X_out3,obs_center,R_plot,i)

subplot(2,2,4)
plotting(X_out4,obs_center,R_plot,i)
    pause(0.001)
    if i~=N
        clf
    end
end

%%
figure(1)
% [X_out,u_out]=fetchOutputs(f2);
R_plot=[2*R_obs-R_agent,R_agent,R_agent,R_agent,R_agent];
R=2*R_plot;
R_plot=R_plot(1:5);
N=length(X_out(1,:));
Num_agen=4;

hold on

for i=1:N
    for j=1:Num_agen
        hold on
        if j==1
            plot(X_out((2*j-1),1:i),X_out((2*j),1:i),'r.')
        elseif j==2
            plot(X_out((2*j-1),1:i),X_out((2*j),1:i),'g.')
        elseif j==3
            plot(X_out((2*j-1),1:i),X_out((2*j),1:i),'b.')
        elseif j==4
            plot(X_out((2*j-1),1:i),X_out((2*j),1:i),'c.')
        end

        %plot(X0((2*i-1),:),X0((2*i),:),'.')
    end

    obs_center=[obs_center(1,:); ...
        X_out(1:2,i)'; ...
        X_out(3:4,i)'; ...
        X_out(5:6,i)'; ...
        X_out(7:8,i)';];

    obs_num=length(R_plot);
    theta=linspace(0,2*pi,201);
    xlim([-5 50])
    ylim([-5 50])
    x_theta=R_plot(1)*cos(theta);
    y_theta=R_plot(1)*sin(theta);
    plot(obs_center(1,1)+1.5*x_theta,obs_center(1,2)+1.5*y_theta,'r')
    for j=1:obs_num
        hold on
        x_theta=R_plot(j)*cos(theta);
        y_theta=R_plot(j)*sin(theta);
        plot(obs_center(1,1)+x_theta,obs_center(1,2)+y_theta,'r')
        if j==1+1
            plot(obs_center(j,1)+x_theta,obs_center(j,2)+y_theta,'r')
        elseif j==2+1
            plot(obs_center(j,1)+x_theta,obs_center(j,2)+y_theta,'g')
        elseif j==3+1
            plot(obs_center(j,1)+x_theta,obs_center(j,2)+y_theta,'b')
        elseif j==4+1
            plot(obs_center(j,1)+x_theta,obs_center(j,2)+y_theta,'c')
        end
    end
    hold on 
    plot([X_out(1,i),X_out(3,i)],[X_out(2,i),X_out(4,i)]); % 1 2
    plot([X_out(1,i),X_out(7,i)],[X_out(2,i),X_out(8,i)]); % 1 4
    plot([X_out(5,i),X_out(3,i)],[X_out(6,i),X_out(4,i)]); % 3 2
    plot([X_out(5,i),X_out(7,i)],[X_out(6,i),X_out(8,i)]); % 3 4
    pause(0.03)
    if i~=N
        clf
    end


end










%% utility funcitons


function ss_max=obs_checker(X_out,Num_agen,obs_center,R)
    
N=length(X_out(1,:));
    for i=1:N

        obs_center=[obs_center(1,:); ...
            X_out(1:2,i)'; ...
            X_out(3:4,i)'; ...
            X_out(5:6,i)'; ...
            X_out(7:8,i)'];
        obs_num=length(obs_center(:,1));
        for j=1:Num_agen
            countk=1;
            for k=1:obs_num
                if k==j+1
                    continue
                end
                ss((countk-1)*(N)+i,j)=R(k)-norm(X_out((2*j-1):(2*j),i)-obs_center(k,:)',2);
                countk=countk+1;
            end
        end

    end

    ss_max=max(ss);

end

function [last_cost,X_out]=traj_gen(Agent_num,Ad,Bd,X,X0,u,R_plot,obs_center,final,Laplacian)


if Agent_num==1
    X_true=[X(7:8,:);X(3:4,:)];
    X=[X(7:8,:);X(1:2,:);X(3:4,:)];
    u=[u(7:8,:);u(1:2,:);u(3:4,:)];
    final=[final(7:8,:);final(1:2,:);final(3:4,:)];
    X0=[X0(7:8,:);X0(1:2,:);X0(3:4,:)];
    
elseif Agent_num==2
    X_true=[X(1:2,:);X(5:6,:)];
    X=[X(1:2,:);X(3:4,:);X(5:6,:)];
    u=[u(1:2,:);u(3:4,:);u(5:6,:)];
    final=[final(1:2,:);final(3:4,:);final(5:6,:)];
    X0=[X0(1:2,:);X0(3:4,:);X0(5:6,:)];

elseif Agent_num==3
    X_true=[X(1:2,:);X(5:6,:)];
    X=[X(3:4,:);X(5:6,:);X(7:8,:)];
    u=[u(3:4,:);u(5:6,:);u(7:8,:)];
    final=[final(3:4,:);final(5:6,:);final(7:8,:)];
    X0=[X0(3:4,:);X0(5:6,:);X0(7:8,:)];

elseif Agent_num==4
    X_true=[X(5:6,:);X(1:2,:)];
    X=[X(5:6,:);X(7:8,:);X(1:2,:)];
    u=[u(5:6,:);u(7:8,:);u(1:2,:)];
    final=[final(5:6,:);final(7:8,:);final(1:2,:)];
    X0=[X0(5:6,:);X0(7:8,:);X0(1:2,:)];
end

obs_center=[12,90; ...
    X(1:2,1)'; ...
    X(3:4,1)'; ...
    X(5:6,1)'];




Num_agen=length(X(:,1))/2;

% for ii=1:Num_agen
%     for jj=1:2
%         final_trans(ii,jj)=final(2*(ii-1)+jj,1);
%     end
% end
% final_centeroid=[sum(final_trans(:,1))/Num_agen,sum(final_trans(:,2))/Num_agen];


R=2*R_plot(1:4);
N=length(X(1,:));
obs_num=length(R);
r_default=0.5;
r=0.5;

lambda=10000;


figure(1)

% kp = 2.5;
% kc = 0.8;


hold on

theta=linspace(0,2*pi,201);
for j=1:obs_num
    x_theta=R_plot(j)*cos(theta);
    y_theta=R_plot(j)*sin(theta);
    plot(obs_center(j,1)+x_theta,obs_center(j,2)+y_theta)
end
Linear_cost=zeros(1,200);
for iteration=1:10



    % V_des=zeros(2*Num_agen,N-1);
    % X_trans=zeros(Num_agen,2);
    % for i=1:N-1
    % 
    %     for ii=1:Num_agen
    %         for jj=1:2
    %             X_trans(ii,jj)=X(2*(ii-1)+jj,i);
    %         end
    %     end
    % 
    %     centroid = (ones(Num_agen,Num_agen) * reshape(X_trans,[4,2]))/4;
    %     centroid_diff = centroid-[final_centeroid(1) final_centeroid(2);final_centeroid(1) final_centeroid(2);final_centeroid(1) final_centeroid(2);final_centeroid(1) final_centeroid(2)];
    % 
    %     V_des_trans = Laplacian*(final_trans-X_trans);
    % 
    %     V_des_trans=kp*V_des_trans - centroid_diff*kc;
    % 
    %     for ii=1:Num_agen
    %         for jj=1:2
    %             if V_des_trans(ii,jj)>=2
    %                 V_des_trans(ii,jj)=2;
    %             elseif V_des_trans(ii,jj)<=-2
    %                 V_des_trans(ii,jj)=-2;
    %             end
    %         end
    %     end
    %     V_des(:,i)=reshape(V_des_trans',[8,1]);
    % end


    cvx_solver SDPT3
    cvx_precision best

    cvx_begin quiet

    variable w(2,N-1)
    variable v(2,N-1)
    variable d(2,N)
    variable s(N*(obs_num-1),1)

    % minimize (  100*norm((u+w-V_des),1) + norm((X+d-X0),1) + ...
    %     1000*lambda*sum(sum(abs(v)))  + lambda*(   sum(sum(max(s,0)))   ))

    minimize (  100*norm((u(3:4,:)+w),1) + norm((X(3:4,:)+d-X0(3:4,:)),1) +  ...
        1000*lambda*sum(sum(abs(v)))  + 1000*lambda*(   sum(sum(max(s,0)))   ))

    subject to
    cvx_precision best

    E=eye(2);
    d(:,1)==zeros(2,1);





    for i=1:N-1

        obs_center=[obs_center(1,:); ...
            X(1:2,i)'; ...
            X(3:4,i)'; ...
            X(5:6,i)'];





        j=2;
        -r<=w(:,i)<=r;



        X((2*j-1):(2*j),i+1)+d(:,i+1)== ...
            (Ad*X((2*j-1):(2*j),i)+Ad*d(:,i))+ ...
            (Bd*u((2*j-1):(2*j),i)+Bd*w(:,i))+E*v(:,i);



        countk=1;

        for k=1:obs_num
            if k==j+1
                continue
            end
            s((countk-1)*(N)+i,1)>=0;
            2*R(k)-norm(X((2*j-1):(2*j),i)-obs_center(k,:)',2)- ...
                (X((2*j-1):(2*j),i)-obs_center(k,:)')'*(X((2*j-1):(2*j),i)+d(:,i)-obs_center(k,:)') ...
                /norm(X((2*j-1):(2*j),i)-obs_center(k,:)',2)<=s((countk-1)*(N)+i,1);
            countk=countk+1;
        end


    end

    X(3:4,N)+d(:,N)==final(3:4);

    cvx_end

    %



    rho0 = 0.03;
    rho1 = 0.25;
    rho2 = 0.75;

    Linear_cost(iteration)=(  100*norm((u(3:4,:)+w),1) + norm((X(3:4,:)+d-X0(3:4,:)),1) +  ...
        1000*lambda*sum(sum(abs(v)))  + 1000*lambda*(   sum(sum(max(s,0)))   ));


    if iteration >= 2
        delta_L = (Linear_cost(iteration) - Linear_cost(iteration-1)) / Linear_cost(iteration);
    else
        delta_L = 1;
    end

    if Linear_cost(iteration)<=10000
        if abs(delta_L) <= rho0
            r = max(r, 0.05);
            X(3:4,:) = X(3:4,:) + d;
            u(3:4,:) = u(3:4,:) + w;
        elseif abs(delta_L) <= rho1
            r = r/1.2;
            X(3:4,:) = X(3:4,:) + d;
            u(3:4,:) = u(3:4,:) + w;
        elseif abs(delta_L) <= rho2
            r = r / 1.5;
            X(3:4,:) = X(3:4,:) + d;
            u(3:4,:) = u(3:4,:) + w;
        else
            X(3:4,:) = X(3:4,:) + d;
            u(3:4,:) = u(3:4,:) + w;
            r = r / 1.6;
        end
    else
        r = r / 1.1;
        X(3:4,:) = X(3:4,:) + d;
        u(3:4,:) = u(3:4,:) + w;
    end
    r;
    hold on
    for i=1:Num_agen
        hold on
        plot(X((2*i-1),:),X((2*i),:),'.')
    end
    ss=0;

    xlim([-5 30])
    ylim([-5 30])

    for i=1:N

        obs_center=[obs_center(1,:); ...
            X(1:2,i)'; ...
            X(3:4,i)'; ...
            X(5:6,i)'];
        for j=1:Num_agen
            countk=1;
            for k=1:obs_num
                if k==j+1
                    continue
                end
                ss((countk-1)*(N)+i,j)=R(k)-norm(X((2*j-1):(2*j),i)-obs_center(k,:)',2);
                countk=countk+1;
            end
        end

    end

    ss_max=max(ss);

    max(max(ss));



    if max(max(ss))<0 && iteration>40
        break;
    end
Linear_cost(iteration)
    if  Linear_cost(iteration)<=2000 && iteration>5
        X_out=X;
        u_out=u;
        last_cost=Linear_cost(iteration);
        break;
    end
    pause(0.01)
    
end
X_out=X;
u_out=u;
last_cost=Linear_cost(iteration);
clf
end



function plotting(X,obs_center,R_plot,I)
R_plot=R_plot(1:4);
N=length(X(1,:));
Num_agen=3;

hold on



for i=I:I
    for j=1:Num_agen
        hold on
        if j==1
            plot(X((2*j-1),i),X((2*j),i),'r.')
        elseif j==2
            plot(X((2*j-1),i),X((2*j),i),'g.')
        elseif j==3
            plot(X((2*j-1),i),X((2*j),i),'b.')
        end


        %plot(X0((2*i-1),:),X0((2*i),:),'.')
    end
    
    obs_center=[obs_center(1,:); ...
        X(1:2,i)'; ...
        X(3:4,i)'; ...
        X(5:6,i)'];

    obs_num=length(R_plot);
    theta=linspace(0,2*pi,201);
    xlim([-5 50])
    ylim([-5 50])
    x_theta=R_plot(1)*cos(theta);
    y_theta=R_plot(1)*sin(theta);
    plot(obs_center(1,1)+1.5*x_theta,obs_center(1,2)+1.5*y_theta,'r')
    for j=1:obs_num
        hold on
        x_theta=R_plot(j)*cos(theta);
        y_theta=R_plot(j)*sin(theta);
        plot(obs_center(1,1)+x_theta,obs_center(1,2)+y_theta,'r')
        if j==1+1
            plot(obs_center(j,1)+x_theta,obs_center(j,2)+y_theta,'r')
        elseif j==2+1
            plot(obs_center(j,1)+x_theta,obs_center(j,2)+y_theta,'g')
        elseif j==3+1
            plot(obs_center(j,1)+x_theta,obs_center(j,2)+y_theta,'b')
        elseif j==4+1
            plot(obs_center(j,1)+x_theta,obs_center(j,2)+y_theta,'c')
        end
    end




end

end






