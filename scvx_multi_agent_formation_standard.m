clc
clear all
clf

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
for t=0:Ts:15
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
kp = 2.5;
kc = 0.8;
Laplacian = - ones(Num_agen) + Num_agen*eye(Num_agen);
%
%
final = [20;25 ;  20;20 ; 25;20   ;  25;25 ];
final = [20;25 ;  20;20 ; 25;20   ;  25;25 ];
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
%     V_des(:,i)=reshape(V_des_trans',[8,1]);
% end




%%
R_obs=2;
R_agent=1.5;

obs_center=[12,9; ...
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
Linear_cost=zeros(1,200);
for iteration=1:60



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
    end


    cvx_solver SDPT3
    cvx_precision best

    cvx_begin

    variable w(2*Num_agen,N-1)

    variable v(2*Num_agen,N-1)
    variable d(2*Num_agen,N)
    variable s(N*(obs_num-1),Num_agen)
    %variable center(N,2)

    %variable s_pos(N)

    minimize (  100*norm((u+w-V_des),1) + norm((X+d-X0),1) + ...
        1000*lambda*sum(sum(abs(v)))  + lambda*(   sum(sum(max(s,0)))   ))

    subject to
    cvx_precision best
    %cvx_precision low
    E=eye(2);
    d(:,1)==zeros(2*Num_agen,1);

    %center(1,:)==[0,0];



    for i=1:N-1

        obs_center=[12,9; ...
            X(1:2,i)'; ...
            X(3:4,i)'; ...
            X(5:6,i)'; ...
            X(7:8,i)'];




        for j=1:Num_agen


            -r<=w((2*j-1):(2*j),i)<=r;



            X((2*j-1):(2*j),i+1)+d((2*j-1):(2*j),i+1)== ...
                (Ad*X((2*j-1):(2*j),i)+Ad*d((2*j-1):(2*j),i))+ ...
                (Bd*u((2*j-1):(2*j),i)+Bd*w((2*j-1):(2*j),i))+E*v((2*j-1):(2*j),i);

 

            countk=1;

            for k=1:obs_num
                if k==j+1
                    continue
                end
                s((countk-1)*(N)+i,j)>=0;
                2*R(k)-norm(X((2*j-1):(2*j),i)-obs_center(k,:)',2)- ...
                    (X((2*j-1):(2*j),i)-obs_center(k,:)')'*(X((2*j-1):(2*j),i)+d((2*j-1):(2*j),i)-obs_center(k,:)') ...
                    /norm(X((2*j-1):(2*j),i)-obs_center(k,:)',2)<=s((countk-1)*(N)+i,j);
                countk=countk+1;
            end
        end

    end

    X(:,N)+d(:,N)==[20;25 ;  20;20 ; 25;20   ;  25;25 ];

    cvx_end

    %



    rho0 = 0.03;
    rho1 = 0.25;
    rho2 = 0.75;

    Linear_cost(iteration)=( 100*norm((u+w-V_des),1) + norm((X+d-X0),1) + ...
        1000*lambda*sum(sum(abs(v)))  + lambda*(   sum(sum(max(s,0)))   ));

    if iteration >= 2
        delta_L = (Linear_cost(iteration) - Linear_cost(iteration-1)) / Linear_cost(iteration);
    else
        delta_L = 1;
    end

    if Linear_cost(iteration)<=10000
        if abs(delta_L) <= rho0
            r = max(r, 0.05);
            X = X + d;
            u = u + w;
        elseif abs(delta_L) <= rho1
            r = r/1.2;
            X = X + d;
            u = u + w;
        elseif abs(delta_L) <= rho2
            r = r / 1.5;
            X = X + d;
            u = u + w;
        else
            X = X + d;
            u = u + w;
            r = r / 1.6;
        end
    else
        r = r / 1.1;
        X = X + d;
        u = u + w;
    end
    r
    hold on
    for i=1:Num_agen
        hold on
        plot(X((2*i-1),:),X((2*i),:),'.')
    end
    ss=0;



    for i=1:N

        obs_center=[12,9; ...
            X(1:2,i)'; ...
            X(3:4,i)'; ...
            X(5:6,i)'; ...
            X(7:8,i)'];
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





    if max(max(ss))<0 && iteration>40
        break;
    end
    pause(0.01)
end


%%
figure(2)
hold on
clf
for i=1:N
    for j=1:Num_agen
        hold on
        if j==1
            plot(X((2*j-1),i),X((2*j),i),'r.')
        elseif j==2
            plot(X((2*j-1),i),X((2*j),i),'g.')
        elseif j==3
            plot(X((2*j-1),i),X((2*j),i),'b.')
        elseif j==4
            plot(X((2*j-1),i),X((2*j),i),'c.')
        end

        %plot(X0((2*i-1),:),X0((2*i),:),'.')
    end

    obs_center=[12,9; ...
        X(1:2,i)'; ...
        X(3:4,i)'; ...
        X(5:6,i)'; ...
        X(7:8,i)'];


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

    pause(0.05)
    %clf


end









%% utility funcitons


function XX=stack_vec(X,u)
XX=reshape(X(:,1:end-1),[length((X(:,1)))*length(X(1,:))-4,1]);
uu=reshape(u,[length((u(:,1)))*length(u(1,:)),1]);
XX=[XX;uu];
end


function [X,u]=unstack_vec(XX,uu)
X=reshape(XX,[4,length(XX)/4]);
u=reshape(uu,[2,length(uu)/2]);
end