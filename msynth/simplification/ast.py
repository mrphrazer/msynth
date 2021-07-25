from builtins import map
from miasm.ir.translators.translator import Translator
from miasm.expression.expression import Expr, ExprCond, ExprSlice, ExprOp, ExprCompose, ExprMem, ExprAssign


class AbstractSyntaxTreeTranslator(Translator): # type: ignore
    """
    Translates a Miasm expression to an abstract syntax tree (AST).

    Since many operations in Miasm have no fixed arity, we enforce 
    a fixed arity for expressions to have deeper trees.

    Example: (x + y + z + z) becomes (x + (y + (z + z))).
    """

    # Implemented language
    __LANG__ = "AST"

    def from_ExprId(self, expr: Expr) -> Expr:
        """
        Translates an ExprId.
        
        Args:
            expr: Expression to translate.
        
        Returns:
            Expression as ExprId.
        """
        return expr

    def from_ExprInt(self, expr: Expr) -> Expr:
        """
        Translates an ExprInt.
        
        Args:
            expr: Expression to translate.
        
        Returns:
            Expression as ExprInt.
        """
        return expr

    def from_ExprLoc(self, expr: Expr) -> Expr:
        """
        Translates an ExprLoc.
        
        Args:
            expr: Expression to translate.
        
        Returns:
            Expression as ExprLoc.
        """
        return expr

    def from_ExprCond(self, expr: Expr) -> Expr:
        """
        Translates an ExprCond.
        
        Args:
            expr: Expression to translate.
        
        Returns:
            Expression as ExprCond.
        """
        return ExprCond(self.from_expr(expr.cond),
                        self.from_expr(expr.src1),
                        self.from_expr(expr.src2))

    def from_ExprSlice(self, expr: Expr) -> Expr:
        """
        Translates an ExprSlice.
        
        Args:
            expr: Expression to translate.
        
        Returns:
            Expression as ExprSlice.
        """
        return ExprSlice(self.from_expr(expr.arg),
                         expr.start,
                         expr.stop)

    def from_ExprOp(self, expr: Expr) -> Expr:
        """
        Translates an ExprOp.
        
        Args:
            expr: Expression to translate.
        
        Returns:
            Expression as ExprOp.
        """
        args = list(map(self.from_expr, expr.args))
        res = args[0]

        if len(args) > 1:
            for arg in args[1:]:
                res = ExprOp(expr.op, res, self.from_expr(arg))
        else:
            res = ExprOp(expr.op, res)
        return res

    def from_ExprCompose(self, expr: Expr) -> Expr:
        """
        Translates an ExprCompose.
        
        Args:
            expr: Expression to translate.
        
        Returns:
            Expression as ExprCompose.
        """
        args = [self.from_expr(arg) for arg in expr.args]
        result = args[0]
        for arg in args[1:]:
            result = ExprCompose(result, arg)
        return result

    def from_ExprAssign(self, expr: Expr) -> Expr:
        """
        Translates an ExprAssign.
        
        Args:
            expr: Expression to translate.
        
        Returns:
            Expression as ExprAssign.
        """
        return ExprAssign(self.from_expr(expr.dst),
                          self.from_expr(expr.src))

    def from_ExprMem(self, expr: Expr) -> Expr:
        """
        Translates an ExprMem.
        
        Args:
            expr: Expression to translate.
        
        Returns:
            Expression as ExprMem.
        """
        return ExprMem(self.from_expr(expr.ptr), expr.size)


# Register the class
Translator.register(AbstractSyntaxTreeTranslator)
