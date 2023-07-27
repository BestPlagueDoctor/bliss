# Geubelle research group
# Authors:  Qibang Liu (qibang@illinois.edu)
#           Michael Zakoworotny (mjz7@illinois.edu)
#           Philippe Geubelle (geubelle@illinois.edu)
#           Aditya Kumar (aditya.kumar@ce.gatech.edu)
#
# Contains a library of commonly used cure kinetics functions
#

from abc import ABC, abstractmethod


class Kinetics_Model(ABC):
    """
    Base class for cure kinetics models
    """

    def __init__(self, KMexpression, para):
        """
        Initializes the cure kinetics model

        :param KMexpression: Expression of the reaction model in latex, required for generating the report
        :type KMexpression: string
        :param para: String in html format to put the parameter in the report table, required for generating the report
        :type para: string    
      
        """
        super().__init__()
        self.KMexpression = KMexpression
        self.para = para

    @abstractmethod
    def eval(self, alpha):
        pass


class First_Order(Kinetics_Model):
    """
    First order reaction model of the form:

    .. math:: \\frac{d\\alpha}{dt} = A\\exp(\\frac{-E}{RT})g(\\alpha)

    .. math:: g(\\alpha) = (1 - \\alpha)
    """

    def __init__(self):
        """
        No parameters needed
        """
        # str - expression of the reaction model in latex
        KMexpression = "          g(\\alpha)= (1-\\alpha),\n"
        # str - string in html format to put the parameter in the report table
        para = None

        super().__init__(KMexpression, para)

    def eval(self, alpha):
        """
        Evaluate cure kinetics

        :param alpha: The function with values of alpha to evalue g(alpha)
        :type alpha: dolfin function
        """
        g = 1 - alpha
        return g


class Nth_Order(Kinetics_Model):
    """
    nth order reaction model of the form:

    .. math:: \\frac{d\\alpha}{dt} = A\\exp(\\frac{-E}{RT})g(\\alpha)

    .. math:: g(\\alpha) = (1 - \\alpha)^n
    """

    def __init__(self, n):
        """
        :param n: Order of reaction
        :type n: float   
        """
        self.n = n
        # super().__init__()
        # str - expression of the reaction model in latex
        KMexpression = "          g(\\alpha)= (1-\\alpha)^n,\n"

        para_tab = ('          <tr>\n')
        para_tab += ('            <td>\( n \)</td>\n')
        para_tab += ('            <td>Order of reaction</td>\n')
        para_tab += ('            <td>%.3f</td>\n' % (self.n))
        para_tab += ('          </tr>\n')
        # self.para = para_tab

        super().__init__(KMexpression, para_tab)

    def eval(self, alpha):
        """
        Evaluate cure kinetics

        :param alpha: The function with values of alpha to evalue g(alpha)
        :type alpha: dolfin function
        """
        g = (1 - alpha)**self.n
        return g


class Prout_Tompkins(Kinetics_Model):
    """
    Prout Tompkins reaction model of the form:

    .. math:: \\frac{d\\alpha}{dt} = A\\exp(\\frac{-E}{RT})g(\\alpha)

    .. math:: g(\\alpha) = (1 - \\alpha)^n \\alpha^m

    """

    def __init__(self, n, m):
        """
        :param n: Order of reaction
        :type n: float
        :param m: Order of reaction
        :type m: float
        """
        self.n = n
        self.m = m
        # str - expression of the reaction model in latex
        KMexpression = "          g(\\alpha)=\\alpha^m (1-\\alpha)^n,\n"
        # m
        para_tab = ('          <tr>\n')
        para_tab += ('            <td>\( m \)</td>\n')
        para_tab += ('            <td>Order of reaction</td>\n')
        para_tab += ('            <td>%.3f</td>\n' % (self.m))
        para_tab += ('          </tr>\n')
        # n
        para_tab += ('          <tr>\n')
        para_tab += ('            <td>\( n \)</td>\n')
        para_tab += ('            <td>Order of reaction</td>\n')
        para_tab += ('            <td>%.3f</td>\n' % (self.n))
        para_tab += ('          </tr>\n')
        # self.para = para_tab
        super().__init__(KMexpression, para_tab)

    def eval(self, alpha):
        """
        Evaluate cure kinetics
        
        :param alpha: The function with values of alpha to evalue g(alpha)
        :type alpha: dolfin function
        """
        g = (1 - alpha)**self.n * alpha**self.m
        return g


class Prout_Tompkins_Diffusion(Kinetics_Model):
    """
    Prout Tompkins with diffusion term reaction model of the form:

    .. math:: \\frac{d\\alpha}{dt} = A\\exp(\\frac{-E}{RT})g(\\alpha)

    .. math:: g(\\alpha) =\\frac{(1 - \\alpha)^n \\alpha^m}{\\exp(Ca(\\alpha - \\alpha_c))}

    """

    def __init__(self, n, m, Ca, alpha_c):
        """
        :param n: Order of reaction
        :type n: float

        :param m: Order of reaction
        :type m: float

        :param Ca: Diffusion constant
        :type Ca: float

        :param alpha_c: Critical conversion at onset of diffusion dominance
        :type alpha_c: float

        """
        self.n = n
        self.m = m
        self.Ca = Ca
        self.alpha_c = alpha_c
        # super().__init__()

        KMexpression = "          g(\\alpha)= (1-\\alpha)^n \\alpha^m \\frac{1}{1+\\exp \\left[C\\left(\\alpha-\\alpha_c\\right)\\right]},\n"
        # m
        para_tab = ('          <tr>\n')
        para_tab += ('            <td>\( m \)</td>\n')
        para_tab += ('            <td>Order of reaction</td>\n')
        para_tab += ('            <td>%.3f</td>\n' % (self.m))
        para_tab += ('          </tr>\n')
        # n
        para_tab += ('          <tr>\n')
        para_tab += ('            <td>\( n \)</td>\n')
        para_tab += ('            <td>Order of reaction</td>\n')
        para_tab += ('            <td>%.3f</td>\n' % (self.n))
        para_tab += ('          </tr>\n')
        # Ca
        para_tab += ('          <tr>\n')
        para_tab += ('            <td>\( C \)</td>\n')
        para_tab += ('            <td>Diffusion constant</td>\n')
        para_tab += ('            <td>%.3f</td>\n' % (self.Ca))
        para_tab += ('          </tr>\n')
        # alpha_c
        para_tab += ('          <tr>\n')
        para_tab += ('            <td>\( \\alpha_c \)</td>\n')
        para_tab += ('            <td>Critical conversion constant</td>\n')
        para_tab += ('            <td>%.3f</td>\n' % (self.alpha_c))
        para_tab += ('          </tr>\n')

        self.para = para_tab

        super().__init__(KMexpression, para_tab)

    def eval(self, alpha):
        """
        Evaluate cure kinetics

        :param alpha: The function with values of alpha to evalue g(alpha)
        :type alpha: dolfin function
        """
        from ufl import exp
        g = (1 - alpha)**self.n * alpha**self.m * 1 / \
            (1 + exp(self.Ca*(alpha - self.alpha_c)))
        return g
