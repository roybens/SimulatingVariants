:[$URL: https://bbpteam.epfl.ch/svn/analysis/trunk/IonChannel/xmlTomod/CreateMOD.c $]
:[$Revision: 1499 $]
:[$Date: 2012-01-28 10:45:44 +0100 (Sat, 28 Jan 2012) $]
:[$Author: rajnish $]
:Comment :
:Reference :Fast rhythmic bursting can be induced in layer 2/3 cortical neurons by enhancing persistent Na+ conductance or by blocking BK channels. J. Neurophysiol., 2003, 89, 909-21
:RBS Modified to https://channelpedia.epfl.ch/ionchannels/213/models/16.mod 
NEURON	{
	SUFFIX Im
	USEION k READ ek WRITE ik
	RANGE gImbar, gIm, ik, a_pre_exp,a_vshift,a_exp_div,b_pre_exp,b_vshift,b_exp_div
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gImbar = 0.00001 (S/cm2) 
	a_pre_exp = 0.02 
	a_vshift = -20
	a_exp_div = 0.2
	b_pre_exp = 0.01
	b_vshift = -43
	b_exp_div = 0.05556
}

ASSIGNED	{
	v	(mV)
	ek	(mV)
	ik	(mA/cm2)
	gIm	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gIm = gImbar*m
	ik = gIm*(v-ek)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
}

PROCEDURE rates(){
	UNITSOFF 
		mAlpha = a_pre_exp/(1+exp((-v+a_vshift)/a_exp_div)) 
		mBeta =  b_pre_exp*exp((-v+ b_vshift)/b_exp_div)
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
}
