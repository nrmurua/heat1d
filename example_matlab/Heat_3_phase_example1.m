close all
clear 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%          STEP 1 : IMPUT DATA           %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% GEOMETRICAL
d=3;                          % Number of the layers
W=[1/3 1/3 1/3];              % Width of the layers
T=1;                          % End time


% PHYSICAL
C=[1 1 1];                      % Heat capacitance
tao_q=[1  1 1];                 % Heat flux phase lags
tao_T=[1 4 4/3];                  % Temperature gradient phase lags
k=[4 1 6];      % Thermal conductivity
alpha=[0.5 0.5];                % Proportionality constants
Knd=[1 1];                     % Knudsen numbers
%functions at the end of file
%heat source on function efe(x,t,Ele) 
%initial conditions on function [psi1 psi2] = initial_conditions(x,Ele)
%bioundary conditions on function boundary_conditions(t)
%extended functions C, k, tau_q and tau_T

% MESH
m=[1000 1000 1000];                      % Layer discretization steps  
N=100;                         % Temporal discretization steps

% ADITIONAL COMPUTATION
L(1)=0;                       % Boundaries and interfaces
for ell=1:d
    L(ell+1)=sum(W(1:ell));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%    STEP 2 : DISCRETIZATION OF DOMAIN   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Index for interfaces and boundaries 
M(1)=1;                           
for ell=1:d
    M(ell+1)=sum(m(1:ell))+1;
end

% Size step discretization of each layer  
for ell=1:d
    DW(ell)=(L(ell+1)-L(ell))/m(ell);
end

% Discretization of the spatial domain [0,L]
for ell=1:d
    for j=M(ell):M(ell+1)
        x(j)=L(ell)+(j-M(ell))*DW(ell);
    end
end

% Size of space step domain [0,L]
for j=1:M(d+1)-1
    Dx(j)=x(j+1)-x(j);
end
Dx(M(d+1))=Dx(M(d+1)-1);

Dt=T/N;              % Size step of time discretization
t=0:Dt:T;            % Time interval discretization



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% STEP 3: EVALUACION OF IC-BC-SOURCE ON MESH %%%
%%%         AND NOTATION PRELIMINAR            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Evaluation of psi_1, psi_2, varphi_1, varphi_2
% Calculate phi_1 and phi_2
% Extend the C, k, tau_q, tau_T and f to all domain.
[u0 Du0] = initial_conditions(x,L);
[varphi_1 varphi_2 Dvarphi_1 Dvarphi_2] = boundary_conditions(t);
phi_1=varphi_1+tao_T(1)*Dvarphi_1;
phi_2=varphi_2+tao_T(d)*Dvarphi_2;
[Cx kx tauqx tauTx] = physical_parameters(x,C,tao_q,tao_T,k,L);
Efe = efe(x,t,L);



%calculus of L^C_j, L^L_j, \Psi^L_j, \Psi^R_j, \mu_j,
%\Psi^{+}_j, \Psi^{-}_j, Z^L_j, Z^U_j, and Z^C_j
EleC=Cx.*(Dt+2*tauqx)/(Dt*Dt);
EleL=2*Cx.*tauqx/(Dt*Dt);
mu=kx./(Dx.*Dx);
PSI_R=1+tauTx/Dt;
PSI_L=-tauTx/Dt;
PSI_pos=0.5+tauTx/Dt;
PSI_neg=0.5-tauTx/Dt;
ZetaL=Cx.*(-Dt+2*tauqx)/(2*Dt*Dt);
ZetaU=Cx.*(Dt+2*tauqx)/(2*Dt*Dt);
ZetaC=ZetaL+ZetaU;
factor_BC0=1+(Dx(M(1))/(alpha(1)*Knd(1)));
factor_BCL=1+(Dx(M(d+1)-1)/(alpha(2)*Knd(2)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%         STEP 4: EVALUACION MATRICES        %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%---------------------------------------------------------%
%       MATRICES hat_A=hA and hat_B=hB and vector hat_s=hs   % 
%       for initialitation step                           % 
%---------------------------------------------------------%


%*************** RIGHT BOUNDARY j=M_0 *******************%
j=M(1);
%-----------DIAGONAL      -----------%
hA(j,j)=EleC(j)+2*mu(j)*factor_BC0*PSI_R(j);
hB(j,j)=EleC(j)-2*mu(j)*factor_BC0*PSI_L(j);    
%-----------UPPER DIAGONAL-----------%
hA(j,j+1)=-2*mu(j)*PSI_R(j);
hB(j,j+1)=2*mu(j)*PSI_L(j);


%*************** INTERIOR j IN I^lay *******************%
for j=M(1)+1:M(d+1)-1
    %-----------UNDER DIAGONAL-----------%
    hA(j,j-1)=-mu(j)*PSI_R(j);
    hB(j,j-1)=mu(j)*PSI_L(j);
    %-----------DIAGONAL      -----------%
    hA(j,j)=EleC(j)+2*mu(j)*PSI_R(j);
    hB(j,j)=EleC(j)-2*mu(j)*PSI_L(j);    
    %-----------UPPER DIAGONAL-----------%
    hA(j,j+1)=-mu(j)*PSI_R(j+1);
    hB(j,j+1)=mu(j)*PSI_L(j+1);
end

%*************** INTERFACES j IN I^int *******************%
for j=M(2):M(d)
    %-----------UNDER DIAGONAL-----------%
    hA(j,j-1)=-2*Dx(j-1)*mu(j-1)*PSI_R(j-1);
    hB(j,j-1)=2*Dx(j-1)*mu(j-1)*PSI_L(j-1);
    %-----------DIAGONAL      -----------%
    hA(j,j)=Dx(j-1)*(EleC(j)+2*mu(j-1)*PSI_R(j-1))+...
        Dx(j)*(EleC(j)+2*mu(j)*PSI_R(j));
    hB(j,j)=Dx(j-1)*(EleC(j)-2*mu(j-1)*PSI_L(j-1))+...
        Dx(j)*(EleC(j)-2*mu(j)*PSI_L(j));
    %-----------UPPER DIAGONAL-----------%
    hA(j,j+1)=-2*Dx(j)*mu(j)*PSI_R(j);
    hB(j,j+1)=2*Dx(j)*mu(j)*PSI_L(j);
end    




%*************** RIGHT BOUNDARY j=M_3 *******************%
j=M(d+1);
%-----------UNDER DIAGONAL-----------%
hA(j,j-1)=-2*mu(j-1)*PSI_R(j-1);
hB(j,j-1)=2*mu(j-1)*PSI_L(j-1);
%-----------DIAGONAL      -----------%
hA(j,j)=EleC(j)+2*mu(j-1)*factor_BCL*PSI_R(j-1);
hB(j,j)=EleC(j)-2*mu(j-1)*factor_BCL*PSI_L(j-1);  






%---------------------------------------------------------%
%       MATRICES A, B and C                               % 
%       for evolution step                                % 
%---------------------------------------------------------%



%*************** RIGHT BOUNDARY j=M_0 *******************%
j=M(1);
%-----------DIAGONAL      -----------%
A(j,j)=ZetaU(j)+mu(j)*factor_BC0*PSI_pos(j);
B(j,j)=ZetaC(j)-mu(j)*factor_BC0;
C(j,j)=-(ZetaL(j)+mu(j)*factor_BC0*PSI_neg(j));    
%-----------UPPER DIAGONAL-----------%
A(j,j+1)=-mu(j)*PSI_pos(j);
B(j,j+1)=mu(j);
C(j,j+1)=mu(j)*PSI_neg(j);



%*************** INTERIOR j IN I^lay *******************%
for j=M(1)+1:M(d+1)-1
    %-----------UNDER DIAGONAL-----------%
    A(j,j-1)=-0.5*mu(j)*PSI_pos(j);
    B(j,j-1)=0.5*mu(j);
    C(j,j-1)=0.5*mu(j)*PSI_neg(j);
    %-----------DIAGONAL      -----------%
    A(j,j)=ZetaU(j)+mu(j)*PSI_pos(j);
    B(j,j)=ZetaC(j)-mu(j);
    C(j,j)=-(ZetaL(j)+mu(j)*PSI_neg(j));    
    %-----------UPPER DIAGONAL-----------%
    A(j,j+1)=-0.5*mu(j)*PSI_pos(j);
    B(j,j+1)=0.5*mu(j);
    C(j,j+1)=0.5*mu(j)*PSI_neg(j);
end

%*************** INTERFACES j IN I^int *******************%
for j=M(2):M(d)
    %-----------UNDER DIAGONAL-----------%
    A(j,j-1)=-Dx(j-1)*mu(j-1)*PSI_pos(j-1);
    B(j,j-1)=Dx(j-1)*mu(j-1);
    C(j,j-1)=Dx(j-1)*mu(j-1)*PSI_neg(j-1);
    %-----------DIAGONAL      -----------%
    A(j,j)=Dx(j-1)*(ZetaU(j)+mu(j-1)*PSI_pos(j-1))+...
        Dx(j)*(ZetaU(j)+mu(j)*PSI_pos(j));    
    B(j,j)=Dx(j-1)*(ZetaC(j)-mu(j-1))+Dx(j)*(ZetaC(j)-mu(j));
    C(j,j)=-(Dx(j-1)*(ZetaL(j)+mu(j-1)*PSI_neg(j-1))+...
        Dx(j)*(ZetaL(j)+mu(j)*PSI_neg(j)));
    %-----------UPPER DIAGONAL-----------%
    A(j,j+1)=-Dx(j)*mu(j)*PSI_pos(j);
    B(j,j+1)=Dx(j)*mu(j);
    C(j,j+1)=Dx(j)*mu(j)*PSI_neg(j);
end






%*************** RIGHT BOUNDARY j=M_3 *******************%
j=M(d+1);
%-----------UNDER DIAGONAL-----------%
A(j,j-1)=-mu(j-1)*PSI_pos(j-1);
B(j,j-1)=mu(j-1);
C(j,j-1)=mu(j-1)*PSI_neg(j-1);
%-----------DIAGONAL      -----------%
A(j,j)=ZetaU(j)+mu(j-1)*factor_BCL*PSI_pos(j-1); 
B(j,j)=ZetaC(j)-mu(j-1)*factor_BCL;
C(j,j)=-(ZetaL(j)+mu(j-1)*factor_BCL*PSI_neg(j-1));  




%---------------------------------------------------------%
%       MATRIZ "Numerical source" to  hat_s=NSource(1,:)  %     
%       for initialitation step and   s^n=NSource(n,:)    % 
%       for evolution step                                %
%---------------------------------------------------------%




%*************** RIGHT BOUNDARY j=M_0 *******************%
j=M(1);
Nsource(1,j)=Dt*EleL(j)*Du0(j)+...
    2*mu(j)*(Dx(j)/(alpha(1)*Knd(1)))*phi_1(2)+Efe(2,j);

%*************** INTERIOR j IN I^lay *******************%
for j=2:M(d+1)-1
    Nsource(1,j)=Dt*EleL(j)*Du0(j)+Efe(2,j);
end

%*************** INTERFACES j IN I^int *******************%
for j=M(2):M(d)
  Nsource(1,j)=Dt*(Dx(j-1)+Dx(j))*EleL(j)*Du0(j)+...
      Dx(j-1)*Efe(2,j-1)+Dx(j)*Efe(2,j);
end    



%*************** RIGHT BOUNDARY j=M_3 *******************%
j=M(d+1);
Nsource(1,j)=Dt*EleL(j)*Du0(j)+...
    2*mu(j-1)*(Dx(j-1)/(alpha(2)*Knd(2)))*phi_2(2)+Efe(2,j);



for n=2:N
    %*************** RIGHT BOUNDARY j=M_0 *******************%
    j=M(1);
    Nsource(n,j)=mu(j)*(Dx(j)/(2*alpha(1)*Knd(1)))*...
        (phi_1(n-1)+2*phi_1(n)+phi_1(n+1))+...
        0.25*(Efe(n-1,j)+2*Efe(n,j)+Efe(n+1,j));
    
    %*************** INTERIOR j IN I^lay *******************%
    for j=M(1)+1:M(d+1)-1
        Nsource(n,j)=0.25*(Efe(n-1,j)+2*Efe(n,j)+Efe(n+1,j));
    end
    
    %*************** INTERFACES j IN I^int *******************%
    for j=M(2):M(d)
        FbarL=0.25*(Efe(n-1,j-1)+2*Efe(n,j-1)+Efe(n+1,j-1));
        FbarR=0.25*(Efe(n-1,j)+2*Efe(n,j)+Efe(n+1,j));
        Nsource(n,j)=Dx(j-1)*FbarL+Dx(j)*FbarR;
    end    
    
    %*************** RIGHT BOUNDARY j=M_3 *******************%
    j=M(d+1);
    Nsource(n,j)=mu(j-1)*(Dx(j-1)/(2*alpha(2)*Knd(2)))*...
        (phi_2(n-1)+2*phi_2(n)+phi_2(n+1))+...
        0.25*(Efe(n-1,j-1)+2*Efe(n,j-1)+Efe(n+1,j-1));
end





% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%      STEP 5: INITIALITATION            %%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 

u(1,:)=u0;
hs=Nsource(1,:);
unew=hA\(hB*u0'+hs');   
u(2,:)=unew';



error(1)=norm(hA*unew-(hB*u0'+hs'));

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%     STEP 6:  EVOLUTION                 %%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
for n=2:N-1
    uold1=u(n-1,:);
    uold=u(n,:);
    sn=Nsource(n+1,:);
    unew=A\(B*uold'+C*uold1'+sn');
    error(n)=norm(A*unew-(B*uold'+C*uold1'+sn'));
    u(n+1,:)=unew';
end



% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%    GRAFICA DE RESULTADOS               %%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Parte 1/2: Los siguientes son para testetear que todo va ok!

 uexacta = analytical_solution(x,t,L);
plot(x,u(1,:),'-',x,uexacta(1,:),'.-')
title('Initial condition')
xlabel('space domain')
ylabel('temperature at t=0')
legend('numerical','analytic')


figure
plot(x,u(2,:),'-',x,uexacta(2,:),'.-')
title('First itearation')
xlabel('space domain')
ylabel('temperature at t=Dt')
legend('numerical','analytic')

figure
plot(x,u(N,:),'-',x,uexacta(N,:),'.-')
title('Solution')
xlabel('space domain')
ylabel('temperature at t=T')
legend('numerical','analytic')
% 
% 
% %%% Parte 2/2: Los siguientes son para el ejemplo 1 del articulo.
% 
% figure
% plot(x,u(1,:),'r--',x,uexacta(1,:),'b.','LineWidth', 2,'MarkerSize',10)
% axis([0 1 0 0.8])
% xlabel('$x$','FontSize',16,'Interpreter','latex')
% ylabel('$u(x,0)$','FontSize',16,'Interpreter','latex')
% legend('$u_{\Delta}(x,0)$','$u(x,0)$','Location','south','Interpreter','latex')
% set([gca],'FontName','Times New Roman' );
% set([gca],'FontSize', 16);
% print -depsc2   -r300 example_uno_ic
% 
% figure
% plot(x, u(N,:),'r--',x,uexacta(N,:),'b.','LineWidth', 2,'MarkerSize',10)
% axis([0 1 0 0.8])
% xlabel('$x$','FontSize',16,'Interpreter','latex')
% ylabel('$u(x,T)$','FontSize',16,'Interpreter','latex')
% legend('$u_{\Delta}(x,T)$','$u(x,T)$','Location','south','FontSize',16,'Interpreter','latex')
% set([gca],'FontName','Times New Roman' );
% set([gca],'FontSize', 16);
% print -depsc2   -r300 example_uno_end
% 
% figure
% surf(t,x,uexacta')
% axis([0 1 0 1  0 0.8])
% xlabel('$t$','FontSize',16,'Interpreter','latex')
% ylabel('$x$','FontSize',16,'Interpreter','latex')
% zlabel('$u(x,t)$','FontSize',16,'Interpreter','latex')
% set([gca],'FontName','Times New Roman' );
% set([gca],'FontSize', 16);
% print -depsc2   -r300 example_u_analytic
% 
% figure
% surf(t(1:N)',x',u')
% axis([0 1 0 1  0 0.8])
% xlabel('$t$','FontSize',16,'Interpreter','latex')
% ylabel('$x$','FontSize',16,'Interpreter','latex')
% zlabel('$u_{\Delta}(x,t)$','FontSize',16,'Interpreter','latex')
% set([gca],'FontName','Times New Roman' );
% set([gca],'FontSize', 16);
% print -depsc2   -r300 example_u_numeric



% for n=1:length(t)-1
%     for j=1:length(x)
%         error_full(n,j)=abs(u(n,j)-uexacta(n,j));
%     end
% end
% 
% 
% error_DX=max(max(error_full))
% 
% %EDx=[6.0353e-02;6.0157e-03;3.7103e-03;3.1112e-03];
% EDx=[2.4247e-4;6.0984e-5;1.5268e-5;3.8187e-6];
% orderDx(1)=log2(EDx(2)/EDx(1));
% for i=2:length(EDx)-1
%   orderDx(i)=log2(EDx(i)/EDx(i-1));
% end


  

  % 1/10   
% 1/20  6.0157e-03
% 1/40 3.7103e-03
% 1/80  3.1112e-03



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%       APPENDIX - FUNCTIONS             %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Heat source f(x,t)
function source = efe(x,t,Ele)
    for n=1:length(t)
        for j=1:length(x)
            if x(j)<Ele(2)
                source(n,j)=((-2+9*pi^2)/8)*exp(-t(n)/2)*sin(3*pi*x(j)/4);
            elseif x(j)<Ele(3)  
                source(n,j)=exp(-t(n)/2)*...
                    (((1+9*pi^2)/4)*cos(3*pi*(2*x(j)-1)/4)-sqrt(2)/4);
            else
                source(n,j)=((-2+9*pi^2)/8)*exp(-t(n)/2).*cos(pi*(3*x(j)-1)/4);
            end
        end
    end
end


% unitial conditions psi_1(x)=u(x,0) and psi_2(x)=D_t u(x,0) 
function [psi1 psi2] = initial_conditions(x,Ele)
for j=1:length(x)
    if x(j)<Ele(2)
        psi1(j)=sin(3*pi*x(j)/4);
        psi2(j)=-0.5*psi1(j);
    elseif x(j)<Ele(3)
        psi1(j)=-cos(3*pi*(2*x(j)-1)/4)+sqrt(2);
        psi2(j)=-0.5*psi1(j);
    else
        psi1(j)=cos(pi*(3*x(j)-1)/4);
        psi2(j)=-0.5*psi1(j);
    end
end
end

% boundary conditions phi_1(t)=u(0,t) and phi_2(t)=u(1,t) 
function [varphi1 varphi2 Dvarphi1 Dvarphi2] = boundary_conditions(t)
    varphi1=-(3*pi/8)*exp(-0.5*t);
    varphi2=-(3*pi/8)*exp(-0.5*t);
    Dvarphi1=(3*pi/16)*exp(-0.5*t);
    Dvarphi2=(3*pi/16)*exp(-0.5*t);    
end

% extended functions C, k, tau_q and tau_T 
function [Cx kx tauqx tauTx] = physical_parameters(x,C,taoq,taoT,k,Ele)
    for j=1:length(x)
        if x(j)<Ele(2)
            Cx(j)=C(1);
            kx(j)=k(1); 
            tauqx(j)=taoq(1); 
            tauTx(j)=taoT(1);
        elseif x(j)<Ele(3)
            Cx(j)=C(2);
            kx(j)=k(2); 
            tauqx(j)=taoq(2); 
            tauTx(j)=taoT(2);            
        else
            Cx(j)=C(3);
            kx(j)=k(3); 
            tauqx(j)=taoq(3); 
            tauTx(j)=taoT(3);
        end
    end   
end


% Analytical solution u(x,t)
function usol = analytical_solution(x,t,Ele)
    for n=1:length(t)
        for j=1:length(x)
            if x(j)<Ele(2)
                usol(n,j)=exp(-t(n)/2)*sin(3*pi*x(j)/4);
            elseif x(j)<Ele(3)  
                usol(n,j)=exp(-t(n)/2)*(-cos(3*pi*(2*x(j)-1)/4)+sqrt(2));
            else
                usol(n,j)=exp(-t(n)/2).*cos(pi*(3*x(j)-1)/4);
            end
        end
    end
end



